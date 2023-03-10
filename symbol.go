// Copyright 2023 The Lit Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"math"
	"math/rand"
	"path/filepath"
	"runtime"
	"sort"
	"strings"

	zim "github.com/akhenakh/gozim"
	"github.com/k3a/html2text"
	bolt "go.etcd.io/bbolt"

	"github.com/pointlander/compress"
)

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

// Square is a square markov vector model
type Square [1 << 16][]uint16

// NewSquareRandom makes new square markov vector model
func NewSquareRandom() *Square {
	rnd := rand.New(rand.NewSource(1))
	vectors := &Square{}
	for i := range vectors {
		vectors[i] = make([]uint16, 1<<16)
	}
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
			fmt.Printf("%5d %5d %s\n", m.Alloc/(1024*1024), i, url)
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

// Learn learns a square markov model from data
func (s Square) Learn(data []byte) {
	for i := 5; i < len(data)-4; i++ {
		index := (uint16(data[i-1]) << 8) | uint16(data[i])
		for j := -4; j < -1; j++ {
			a := (uint16(data[i-1-j]) << 8) | uint16(data[i-j])
			if s[index][a] == math.MaxUint16 {
				for k := range s[index] {
					s[index][k] >>= 1
				}
			}
			s[index][a]++

			if s[index&0xff][a] == math.MaxUint16 {
				for k := range s[index&0xff] {
					s[index&0xff][k] >>= 1
				}
			}
			s[index&0xff][a]++
		}
		for j := 2; j < 5; j++ {
			a := (uint16(data[i-1+j]) << 8) | uint16(data[i+j])
			if s[index][a] == math.MaxUint16 {
				for k := range s[index] {
					s[index][k] >>= 1
				}
			}
			s[index][a]++

			if s[index&0xff][a] == math.MaxUint16 {
				for k := range s[index&0xff] {
					s[index&0xff][k] >>= 1
				}
			}
			s[index&0xff][a]++
		}
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
