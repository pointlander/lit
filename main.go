// Copyright 2023 The Lit Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"

	"github.com/pointlander/compress"
	"github.com/pointlander/pagerank"

	zim "github.com/akhenakh/gozim"
	"github.com/k3a/html2text"
	bolt "go.etcd.io/bbolt"
)

const (
	// Order is the order of the markov word vector model
	Order = 9
	// Depth is the depth of the search
	Depth = 2
	// Width is the width of the probability distribution
	Width = 256
)

// Vector is a word vector
type Vector struct {
	Word   string
	Vector []float64
}

// Vectors is a set of word vectors
type Vectors struct {
	List       []Vector
	Dictionary map[string]Vector
}

// NewVectors creates a new word vector set
func NewVectors(file string) Vectors {
	vectors := Vectors{
		Dictionary: make(map[string]Vector),
	}
	in, err := os.Open(file)
	if err != nil {
		panic(err)
	}
	defer in.Close()

	gzipReader, err := gzip.NewReader(in)
	if err != nil {
		panic(err)
	}
	reader := bufio.NewReader(gzipReader)
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
		}
		parts := strings.Split(line, " ")
		values := make([]float64, 0, len(parts)-1)
		for _, v := range parts[1:] {
			n, err := strconv.ParseFloat(strings.TrimSpace(v), 64)
			if err != nil {
				panic(err)
			}
			values = append(values, float64(n))
		}
		sum := 0.0
		for _, v := range values {
			sum += v * v
		}
		length := math.Sqrt(sum)
		for i, v := range values {
			values[i] = v / length
		}
		word := strings.ToLower(strings.TrimSpace(parts[0]))
		vector := Vector{
			Word:   word,
			Vector: values,
		}
		vectors.List = append(vectors.List, vector)
		vectors.Dictionary[word] = vector
		if len(vector.Vector) == 0 {
			fmt.Println(vector)
		}
	}
	return vectors
}

// Entropy calculates entropy
func (v *Vectors) Entropy(input []string) (ax []float64) {
	width := len(v.Dictionary["dog"].Vector)
	length := len(input)
	weights := NewMatrix(width, length)
	for _, word := range input {
		vector := v.Dictionary[word].Vector
		weights.Data = append(weights.Data, vector...)
	}

	l1 := Softmax(Mul(weights, weights))
	l2 := Softmax(Mul(T(weights), l1))
	entropy := Entropy(l2)

	return entropy.Data
}

// Symbols is a set of ordered symbols
type Symbols [Order]uint8

// SymbolVectors are markov symbol vectors
type SymbolVectors map[Symbols]map[byte]uint16

// NewSymbolVectors makes new markov symbol vector model
func NewSymbolVectors() SymbolVectors {
	vectors := make(SymbolVectors)
	data, err := filepath.Abs(*FlagData)
	if err != nil {
		panic(err)
	}
	reader, err := zim.NewReader(data, false)
	if err != nil {
		panic(err)
	}
	var m runtime.MemStats
	i, articles := 0, reader.ListArticles()
	for article := range articles {
		url := article.FullURL()
		if strings.HasSuffix(url, ".html") {
			html, err := article.Data()
			if err != nil {
				panic(err)
			}
			plain := html2text.HTML2Text(string(html))
			runtime.ReadMemStats(&m)
			fmt.Printf("%5d %20d %s\n", m.Alloc/(1024*1024), len(vectors), url)
			vectors.Learn([]byte(plain))
			if i%100 == 0 {
				runtime.GC()
			}
			i++
		}
	}
	fmt.Println("done")
	return vectors
}

// NewSymbolVectorsRandom makes new markov symbol vector model
func NewSymbolVectorsRandom() SymbolVectors {
	rnd := rand.New(rand.NewSource(1))
	vectors := make(SymbolVectors)
	data, err := filepath.Abs(*FlagData)
	if err != nil {
		panic(err)
	}
	reader, err := zim.NewReader(data, false)
	if err != nil {
		panic(err)
	}
	var m runtime.MemStats
	i, length := 0, reader.ArticleCount
	for {
		index := rnd.Intn(int(length))
		if index == 0 {
			continue
		}
		article, err := reader.ArticleAtURLIdx(uint32(index))
		if err != nil {
			continue
		}
		url := article.FullURL()
		if strings.HasSuffix(url, ".html") {
			html, err := article.Data()
			if err != nil {
				panic(err)
			}
			plain := html2text.HTML2Text(string(html))
			runtime.ReadMemStats(&m)
			fmt.Printf("%5d %20d %s\n", m.Alloc/(1024*1024), len(vectors), url)
			vectors.Learn([]byte(plain))
			if i%100 == 0 {
				runtime.GC()
			}
			if i == *FlagScale*1024 {
				break
			}
			i++
		}
	}
	fmt.Println("done")
	return vectors
}

// Learn learns a markov model from data
func (s SymbolVectors) Learn(data []byte) {
	var symbols Symbols
	for i, symbol := range data[:len(data)-Order+1] {
		for j := 0; j < Order-1; j++ {
			symbols := symbols
			for k := 0; k < j; k++ {
				symbols[k] = 0
			}
			vector := s[symbols]
			if vector == nil {
				vector = make(map[byte]uint16, 1)
			}
			if vector[symbol] < math.MaxUint16 {
				vector[symbol]++
			} else {
				for key, value := range vector {
					vector[key] = value >> 1
				}
				vector[symbol]++
			}
			for j := 1; j < Order; j++ {
				if vector[data[i+j]] < math.MaxUint16 {
					vector[data[i+j]]++
				} else {
					for key, value := range vector {
						vector[key] = value >> 1
					}
					vector[data[i+j]]++
				}
			}
			s[symbols] = vector
		}
		for i, value := range symbols[1:] {
			symbols[i] = value
		}
		symbols[Order-1] = symbol
	}
}

// MarkovProbability calculates the markov probability
func MarkovProbability(db *bolt.DB, input []byte) (ax []float64) {
	length := len(input)
	weights := NewMatrix(Width, length-Order+1)
	orders := make([]int, length-Order+1)
	for i := 0; i < length-Order+1; i++ {
		symbol := Symbols{}
		for j := range symbol {
			symbol[j] = input[i+j]
		}
		var decoded [Width]uint16
		found, order := false, 0
		db.View(func(tx *bolt.Tx) error {
			b := tx.Bucket([]byte("markov"))
			for j := 0; j < Order-1; j++ {
				symbol := symbol
				for k := 0; k < j; k++ {
					symbol[k] = 0
				}
				v := b.Get(symbol[:])
				if v != nil {
					found, order = true, j
					index, buffer, output := 0, bytes.NewBuffer(v), make([]byte, 2*Width)
					compress.Mark1Decompress1(buffer, output)
					for key := range decoded {
						decoded[key] = uint16(output[index])
						index++
						decoded[key] |= uint16(output[index]) << 8
						index++
					}
					return nil
				}
			}
			return nil
		})
		if !found {
			orders[i] = 0
			vector := make([]float64, Width)
			weights.Data = append(weights.Data, vector...)
		} else {
			orders[i] = Order - order
			vector, sum := make([]float64, Width), float64(0.0)
			for key, value := range decoded {
				if value == math.MaxUint16 {
					fmt.Println("max value")
				}
				v := float64(value)
				sum += v * v
				vector[key] = v
			}
			length := math.Sqrt(sum)
			for i, v := range vector {
				vector[i] = v / length
			}
			weights.Data = append(weights.Data, vector...)
		}
	}

	probabilities, index := make([]float64, length-Order+1), 0
	for i := 0; i < len(weights.Data); i += Width {
		probabilities[index] = weights.Data[i+int(input[index+Order-1])] * float64(orders[index])
		index++
	}

	return probabilities
}

// SelfEntropy calculates entropy
func SelfEntropy(db *bolt.DB, input []byte) (ax []float64) {
	rnd := rand.New(rand.NewSource(1))
	length := len(input)
	weights := NewMatrix(Width, length-Order+1)
	orders := make([]int, length-Order+1)
	for i := 0; i < length-Order+1; i++ {
		symbol := Symbols{}
		for j := range symbol {
			symbol[j] = input[i+j]
		}
		var decoded [Width]uint16
		found, order := false, 0
		db.View(func(tx *bolt.Tx) error {
			b := tx.Bucket([]byte("markov"))
			for j := 0; j < Order-1; j++ {
				symbol := symbol
				for k := 0; k < j; k++ {
					symbol[k] = 0
				}
				v := b.Get(symbol[:])
				if v != nil {
					found, order = true, j
					index, buffer, output := 0, bytes.NewBuffer(v), make([]byte, 2*Width)
					compress.Mark1Decompress1(buffer, output)
					for key := range decoded {
						decoded[key] = uint16(output[index])
						index++
						decoded[key] |= uint16(output[index]) << 8
						index++
					}
					return nil
				}
			}
			return nil
		})
		if !found {
			orders[i] = Order - 1
			vector, sum := make([]float64, Width), float64(0.0)
			for key := range vector {
				v := rnd.Float64()
				sum += v * v
				vector[key] = v
			}
			length := math.Sqrt(sum)
			for i, v := range vector {
				vector[i] = v / length
			}
			weights.Data = append(weights.Data, vector...)
		} else {
			orders[i] = order
			vector, sum := make([]float64, Width), float64(0.0)
			for key, value := range decoded {
				if value == math.MaxUint16 {
					fmt.Println("max value")
				}
				v := float64(value)
				sum += v * v
				vector[key] = v
			}
			length := math.Sqrt(sum)
			for i, v := range vector {
				vector[i] = v / length
			}
			weights.Data = append(weights.Data, vector...)
		}
	}

	importance := NewMatrix(len(orders), 1)
	for _, order := range orders {
		importance.Data = append(importance.Data, 1/float64(Order-order))
	}

	entropy := make([]float64, 1)
	entropy[0] = SelfEntropyKernel(weights, weights, weights, importance)

	return entropy
}

var (
	// FlagMarkov mode uses markov symbol vectors
	FlagMarkov = flag.Bool("markov", false, "markov symbol vector mode")
	// FlagAttention mode uses markov symbols with attention
	FlagAttention = flag.Bool("attention", false, "markov symbol attention mode")
	// FlagDiffusion is a diffusion based model
	FlagDiffusion = flag.Bool("diffusion", false, "diffusion mode")
	// FlagInput is the input into the markov model
	FlagInput = flag.String("input", "What color is the sky?", "input into the markov model")
	// FlagRandomInput use random input
	FlagRandomInput = flag.Int("randomInput", 0, "random string")
	// FlagPagerank page rank mode
	FlagPageRank = flag.Bool("pagerank", false, "pagerank mode")
	// FlagLearn learn a model
	FlagLearn = flag.Bool("learn", false, "learns a model")
	// FlagData is the path to the training data
	FlagData = flag.String("data", "gutenberg_en_all_2022-04.zim", "path to the training data")
	// FlagRanom select random books from gutenberg for training
	FlagRandom = flag.Bool("random", false, "use random books from gutenberg")
	// FlagScale the scaling factor for the amount of samples
	FlagScale = flag.Int("scale", 8, "the scaling factor for the amount of samples")
)

type Result struct {
	Entropy float64
	Output  []byte
}

func split(pathes []Result) int {
	sum := 0.0
	for _, e := range pathes {
		sum += e.Entropy
	}
	avg, vari := sum/float64(len(pathes)), 0.0
	for _, e := range pathes {
		difference := e.Entropy - avg
		vari += difference * difference
	}
	vari /= float64(len(pathes))

	index, max := 1, 0.0
	for i := 1; i < len(pathes); i++ {
		suma, counta := 0.0, 0.0
		for _, e := range pathes[:i] {
			suma += e.Entropy
			counta++
		}
		avga, varia := suma/counta, 0.0
		for _, e := range pathes[:i] {
			difference := e.Entropy - avga
			varia += difference * difference
		}
		varia /= counta

		sumb, countb := 0.0, 0.0
		for _, e := range pathes[i:] {
			sumb += e.Entropy
			countb++
		}
		avgb, varib := sumb/countb, 0.0
		for _, e := range pathes[i:] {
			difference := e.Entropy - avgb
			varib += difference * difference
		}
		varib /= countb
		gain := vari - (varia + varib)
		if gain > max {
			index, max = i, gain
		}
	}
	return index
}

func markov() {
	db, err := bolt.Open("model.bolt", 0600, nil)
	if err != nil {
		panic(err)
	}
	defer db.Close()

	in := []byte(*FlagInput)
	var search func(depth int, input []byte, done chan Result)
	search = func(depth int, input []byte, done chan Result) {
		pathes := make([]Result, Width)
		for i := 0; i < Width; i++ {
			n := make([]byte, len(input))
			copy(n, input)
			n = append(n, byte(i))
			pathes[i].Output = n
			total := 0.0
			entropy := MarkovProbability(db, n)
			for _, value := range entropy {
				total += value
			}
			pathes[i].Entropy = total
		}
		sort.Slice(pathes, func(i, j int) bool {
			return pathes[i].Entropy > pathes[j].Entropy
		})
		index := split(pathes)
		/*for _, path := range pathes[:index] {
			fmt.Println(path.Entropy,
				strings.Map(func(r rune) rune {
					if unicode.IsPrint(r) {
						return r
					}
					return -1
				}, "("+string(path.Output))+")")
		}*/
		max, output := 0.0, []byte{}
		if depth <= 1 {
			max, output = pathes[0].Entropy, pathes[0].Output
		} else {
			next := make(chan Result, 8)
			for _, path := range pathes[:index] {
				go search(depth-1, path.Output, next)
			}
			for range pathes[:index] {
				result := <-next
				if result.Entropy > max {
					max, output = result.Entropy, result.Output
				}
			}
		}
		done <- Result{
			Entropy: max,
			Output:  output,
		}
	}
	padding := make([]byte, Order-2)
	in = append(padding, in...)
	done := make(chan Result, 8)
	go search(Depth, in, done)
	result := <-done
	fmt.Println(result.Entropy, string(result.Output))
	fmt.Printf("\n")
	for i := 0; i < 128; i++ {
		search(Depth, result.Output, done)
		result = <-done
		fmt.Println(result.Entropy, string(result.Output))
		fmt.Printf("\n")
	}
}

func markovSelfEntropy() {
	db, err := bolt.Open("model.bolt", 0600, nil)
	if err != nil {
		panic(err)
	}
	defer db.Close()

	in := []byte(*FlagInput)
	var search func(depth int, input []byte, done chan Result)
	search = func(depth int, input []byte, done chan Result) {
		pathes := make([]Result, Width)
		for i := 0; i < Width; i++ {
			n := make([]byte, len(input))
			copy(n, input)
			n = append(n, byte(i))
			pathes[i].Output = n
			total := 0.0
			entropy := SelfEntropy(db, n)
			for _, value := range entropy {
				total += value
			}
			pathes[i].Entropy = total
		}
		sort.Slice(pathes, func(i, j int) bool {
			return pathes[i].Entropy < pathes[j].Entropy
		})
		index := split(pathes)
		/*for _, path := range pathes[:index] {
			fmt.Println(path.Entropy,
				strings.Map(func(r rune) rune {
					if unicode.IsPrint(r) {
						return r
					}
					return -1
				}, "("+string(path.Output))+")")
		}*/
		min, output := math.MaxFloat64, []byte{}
		if depth <= 1 {
			min, output = pathes[0].Entropy, pathes[0].Output
		} else {
			next := make(chan Result, 8)
			for _, path := range pathes[:index] {
				go search(depth-1, path.Output, next)
			}
			for range pathes[:index] {
				result := <-next
				if result.Entropy < min {
					min, output = result.Entropy, result.Output
				}
			}
		}
		done <- Result{
			Entropy: min,
			Output:  output,
		}
	}
	padding := make([]byte, Order-2)
	in = append(padding, in...)
	done := make(chan Result, 8)
	go search(Depth, in, done)
	result := <-done
	result.Output = result.Output[:len(result.Output)-Depth+1]
	fmt.Println(result.Entropy, string(result.Output))
	fmt.Printf("\n")
	for i := 0; i < 128; i++ {
		search(Depth, result.Output, done)
		result = <-done
		result.Output = result.Output[:len(result.Output)-Depth+1]
		fmt.Println(result.Entropy, string(result.Output))
		fmt.Printf("\n")
	}
}

func markovSelfEntropyDiffusion() {
	rnd := rand.New(rand.NewSource(1))

	db, err := bolt.Open("model.bolt", 0600, nil)
	if err != nil {
		panic(err)
	}
	defer db.Close()

	in := []byte(*FlagInput)
	if *FlagRandomInput != 0 {
		rnd := rand.New(rand.NewSource(int64(*FlagRandomInput)))
		symbols := []byte("abcdefghijklmnopqrstuvwxyz")
		for i := range in {
			in[i] = symbols[rnd.Intn(len(symbols))]
		}
	}
	var search func(index, depth int, input []byte, done chan Result)
	search = func(idx, depth int, input []byte, done chan Result) {
		pathes := make([]Result, Width)
		for i := 0; i < Width; i++ {
			n := make([]byte, len(input))
			copy(n, input)
			n[idx] = byte(i)
			pathes[i].Output = n
			total := 0.0
			entropy := SelfEntropy(db, n)
			for _, value := range entropy {
				total += value
			}
			pathes[i].Entropy = total
		}
		sort.Slice(pathes, func(i, j int) bool {
			return pathes[i].Entropy < pathes[j].Entropy
		})
		index := split(pathes)
		/*for _, path := range pathes[:index] {
			fmt.Println(path.Entropy,
				strings.Map(func(r rune) rune {
					if unicode.IsPrint(r) {
						return r
					}
					return -1
				}, "("+string(path.Output))+")")
		}*/
		min, output := math.MaxFloat64, []byte{}
		if depth <= 1 {
			min, output = pathes[0].Entropy, pathes[0].Output
		} else {
			next := make(chan Result, 8)
			for _, path := range pathes[:index] {
				go search(idx, depth-1, path.Output, next)
			}
			for range pathes[:index] {
				result := <-next
				if result.Entropy < min {
					min, output = result.Entropy, result.Output
				}
			}
		}
		done <- Result{
			Entropy: min,
			Output:  output,
		}
	}
	padding := make([]byte, Order-2)
	size := len(in)
	in = append(padding, in...)
	done := make(chan Result, 8)
	go search(Order-2+rnd.Intn(size), 1, in, done)
	result := <-done
	fmt.Println(result.Entropy, string(result.Output))
	fmt.Printf("\n")
	for i := 0; i < 512; i++ {
		search(Order-2+rnd.Intn(size), 1, result.Output, done)
		result = <-done
		fmt.Println(result.Entropy, string(result.Output))
		fmt.Printf("\n")
	}
}

func main() {
	flag.Parse()

	if *FlagMarkov {
		markov()
		return
	} else if *FlagAttention {
		markovSelfEntropy()
		return
	} else if *FlagDiffusion {
		markovSelfEntropyDiffusion()
		return
	} else if *FlagPageRank {
		db, err := bolt.Open("model.bolt", 0600, nil)
		if err != nil {
			panic(err)
		}
		defer db.Close()

		lookup := func(symbol Symbols) (found bool, vector []float64) {
			var decoded [Width]uint16
			db.View(func(tx *bolt.Tx) error {
				b := tx.Bucket([]byte("markov"))
				v := b.Get(symbol[:])
				if v != nil {
					found = true
					index, buffer, output := 0, bytes.NewBuffer(v), make([]byte, 2*Width)
					compress.Mark1Decompress1(buffer, output)
					for key := range decoded {
						decoded[key] = uint16(output[index])
						index++
						decoded[key] |= uint16(output[index]) << 8
						index++
					}
					return nil
				}
				return nil
			})
			if !found {
				return found, nil
			}
			vector, sum := make([]float64, Width), float64(0.0)
			for key, value := range decoded {
				v := float64(value)
				sum += v * v
				vector[key] = v
			}
			length := math.Sqrt(sum)
			for i, v := range vector {
				vector[i] = v / length
			}
			return found, vector
		}

		graph := pagerank.NewGraph64()
		for i := 0; i < Width*Width; i++ {
			x := Symbols{}
			x[Order-2] = byte(i >> 8)
			x[Order-1] = byte(i & 0xff)
			found, a := lookup(x)
			if !found {
				continue
			}
			for j := 0; j < Width*Width; j++ {
				y := Symbols{}
				y[Order-2] = byte(j >> 8)
				y[Order-1] = byte(j & 0xff)
				found, b := lookup(y)
				if !found {
					continue
				}
				sum := 0.0
				for k, value := range a {
					sum += value * b[k]
				}
				graph.Link(uint64(i), uint64(j), sum)
			}
		}
		fmt.Println("graph built")
		type Node struct {
			Node int
			Rank float64
		}
		nodes := make([]Node, 0, 8)
		graph.Rank(0.85, 1e-12, func(node uint64, rank float64) {
			nodes = append(nodes, Node{
				Node: int(node),
				Rank: rank,
			})
		})
		fmt.Println("ranking done")
		sort.Slice(nodes, func(i, j int) bool {
			return nodes[i].Rank > nodes[j].Rank
		})
		fmt.Println("sorting done")
		output, err := os.Create("output.txt")
		if err != nil {
			panic(err)
		}
		defer output.Close()
		for _, node := range nodes {
			fmt.Fprintf(output, "%04x %.12f\n", node.Node, node.Rank)
		}
		return
	} else if *FlagLearn {
		var s SymbolVectors
		if *FlagRandom {
			s = NewSymbolVectorsRandom()
		} else {
			s = NewSymbolVectors()
		}

		fmt.Println("done building")
		db, err := bolt.Open("model.bolt", 0666, nil)
		if err != nil {
			panic(err)
		}
		defer db.Close()
		db.Update(func(tx *bolt.Tx) error {
			_, err := tx.CreateBucket([]byte("markov"))
			if err != nil {
				panic(err)
			}
			return nil
		})
		fmt.Println("write file")
		type Pair struct {
			Key   []byte
			Value []byte
		}
		length, count, i, pairs := len(s), 0, 0, [1024]Pair{}
		for key, value := range s {
			k := make([]byte, len(key))
			copy(k, key[:])
			pairs[i].Key = k
			v := make([]uint16, Width)
			for key, value := range value {
				v[key] = value
			}
			index, data := 0, make([]byte, 2*Width)
			for _, value := range v {
				data[index] = byte(value & 0xff)
				index++
				data[index] = byte((value >> 8) & 0xff)
				index++
			}
			pairs[i].Value = data
			delete(s, key)
			i++
			count++
			if i == len(pairs) {
				db.Update(func(tx *bolt.Tx) error {
					b := tx.Bucket([]byte("markov"))
					for _, pair := range pairs {
						buffer := bytes.Buffer{}
						compress.Mark1Compress1(pair.Value, &buffer)
						err := b.Put(pair.Key, buffer.Bytes())
						if err != nil {
							return err
						}
					}
					return nil
				})
				i = 0
				fmt.Printf("%f\n", float64(count)/float64(length))
			}
		}
		if i > 0 {
			db.Update(func(tx *bolt.Tx) error {
				b := tx.Bucket([]byte("markov"))
				for _, pair := range pairs[:i] {
					buffer := bytes.Buffer{}
					compress.Mark1Compress1(pair.Value, &buffer)
					err := b.Put(pair.Key, buffer.Bytes())
					if err != nil {
						return err
					}
				}
				return nil
			})
		}
		fmt.Println("done writing file")
		return
	}

	v := NewVectors("cc.en.300.vec.gz")

	input := []string{"and", "god", "said", "let", "there", "be", "light", "and", "there", "was"}
	type Word struct {
		Word    string
		Entropy float64
	}
	list := make([]Word, 0, 8)
	target := make([]float64, 300)
	entropy := v.Entropy(input)
	for i, e := range entropy {
		for j, value := range v.Dictionary[input[i]].Vector {
			target[j] += e * value
		}
		list = append(list, Word{
			Word:    input[i],
			Entropy: e,
		})
	}
	sort.Slice(list, func(i, j int) bool {
		return list[i].Entropy < list[j].Entropy
	})
	for _, word := range list {
		fmt.Println(word)
	}

	words, words2, words3, max, m, min := []string{}, []string{}, []string{}, 0.0, 0.0, math.MaxFloat64
	for _, w := range v.List {
		ab, aa, bb := 0.0, 0.0, 0.0
		for i, a := range w.Vector {
			b := target[i]
			ab += a * b
			aa += a * a
			bb += b * b
		}
		s := ab / (math.Sqrt(aa) * math.Sqrt(bb))
		if s > max {
			max, words = s, append(words, w.Word)
		}

		entropy := v.Entropy(append(input, w.Word))
		sum := 0.0
		for _, e := range entropy {
			sum += e
		}
		if sum > m {
			m, words2 = sum, append(words2, w.Word)
		}
		if sum < min {
			min, words3 = sum, append(words3, w.Word)
		}
	}
	fmt.Println(words)
	fmt.Println(words2)
	fmt.Println(words3)
}
