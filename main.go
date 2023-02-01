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
	"runtime"
	"sort"
	"strconv"
	"strings"

	"github.com/pointlander/compress"

	zim "github.com/akhenakh/gozim"
	"github.com/k3a/html2text"
	bolt "go.etcd.io/bbolt"
)

const (
	// Order is the order of the markov word vector model
	Order = 9
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
	reader, err := zim.NewReader("/home/andrew/Downloads/gutenberg_en_all_2022-04.zim", false)
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
	reader, err := zim.NewReader("/home/andrew/Downloads/gutenberg_en_all_2022-04.zim", false)
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
		vector := s[symbols]
		if vector == nil {
			vector = make(map[byte]uint16, 1)
		}
		if vector[symbol] < math.MaxUint16 {
			vector[symbol]++
		}
		for j := 1; j < Order; j++ {
			if vector[data[i+j]] < math.MaxUint16 {
				vector[data[i+j]]++
			}
		}
		s[symbols] = vector
		for i, value := range symbols[1:] {
			symbols[i] = value
		}
		symbols[Order-1] = symbol
	}
}

// SelfEntropy calculates entropy
func SelfEntropy(db *bolt.DB, input []byte) (ax []float64) {
	rnd := rand.New(rand.NewSource(1))
	width := 256
	vector := make([]float64, 256)
	length := len(input)
	weights := NewMatrix(width, length-Order+1)
	for i := 0; i < length-Order+1; i++ {
		symbol := Symbols{}
		for j := range symbol {
			symbol[j] = input[i+j]
		}
		var decoded [256]uint16
		found := false
		db.View(func(tx *bolt.Tx) error {
			b := tx.Bucket([]byte("markov"))
			v := b.Get(symbol[:])
			if v != nil {
				found = true
				index, buffer, output := 0, bytes.NewBuffer(v), make([]byte, 512)
				compress.Mark1Decompress1(buffer, output)
				for key := range decoded {
					decoded[key] = uint16(output[index])
					index++
					decoded[key] |= uint16(output[index]) << 8
					index++
				}
			}
			return nil
		})
		if !found {
			for j := 0; j < 256; j++ {
				weights.Data = append(weights.Data, 0)
			}
		} else {
			sum := float64(0.0)
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

	projection := NewRandMatrix(rnd, 256, 256)
	l1 := Softmax(Mul(Normalize(Mul(projection, weights)), weights))
	l2 := Softmax(Mul(T(weights), l1))
	entropy := Entropy(l2)

	return entropy.Data
}

var (
	// FlagMarkov mode use markov symbol vectors
	FlagMarkov = flag.Bool("markov", false, "markov symbol vector mode")
	// FlagLearn learn a model
	FlagLearn = flag.Bool("learn", false, "learns a model")
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

	index, max := 0, 0.0
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

	in := []byte("What color is the sky?")
	//in := []byte("Is the sky blue?")
	var search func(depth int, input []byte, done chan Result)
	search = func(depth int, input []byte, done chan Result) {
		pathes := make([]Result, 256)
		for i := 0; i < 256; i++ {
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
	done := make(chan Result, 8)
	go search(2, in, done)
	result := <-done
	fmt.Println(result.Entropy, string(result.Output))
	fmt.Printf("\n")
	for i := 0; i < 128; i++ {
		search(2, result.Output, done)
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
	}

	if *FlagLearn {
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
			v := make([]uint16, 256)
			for key, value := range value {
				v[key] = value
			}
			index, data := 0, make([]byte, 512)
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
