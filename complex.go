// Copyright 2023 The Lit Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"math"
	"math/cmplx"
	"math/rand"
	"path/filepath"
	"runtime"
	"sort"
	"strings"

	zim "github.com/akhenakh/gozim"
	"github.com/k3a/html2text"
	"github.com/pointlander/compress"
	bolt "go.etcd.io/bbolt"
)

// ComplexSymbols is a set of ordered symbols
type ComplexSymbols [ComplexOrder]uint8

// ComplexSymbolVectors are markov complex symbol vectors
type ComplexSymbolVectors map[ComplexSymbols][]complex64

// NewComplexSymbolVectors makes new markov complex symbol vector model
func NewComplexSymbolVectors() ComplexSymbolVectors {
	rnd := rand.New(rand.NewSource(1))
	vectors := make(ComplexSymbolVectors)
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
			vectors.Learn(rnd, []byte(plain))
			if i%100 == 0 {
				runtime.GC()
			}
			i++
		}
	}
	fmt.Println("done")
	return vectors
}

// NewComplexSymbolVectorsRandom makes new markov complex symbol vector model
func NewComplexSymbolVectorsRandom() ComplexSymbolVectors {
	rnd := rand.New(rand.NewSource(1))
	vectors := make(ComplexSymbolVectors)
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
			vectors.Learn(rnd, []byte(plain))
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
func (s ComplexSymbolVectors) Learn(rnd *rand.Rand, data []byte) {
	const Eta = .1
	var symbols ComplexSymbols
	for i, symbol := range data[:len(data)-Order+1] {
		for j := 0; j < ComplexOrder-1; j++ {
			symbols := symbols
			for k := 0; k < j; k++ {
				symbols[k] = 0
			}
			vector := s[symbols]
			if vector == nil {
				vector = make([]complex64, 0, Width)
				factor := math.Sqrt(2.0 / float64(Width))
				for i := 0; i < Width; i++ {
					vector = append(vector, complex(float32(rnd.NormFloat64()*factor), float32(rnd.NormFloat64()*factor)))
				}
			}
			inputs := make([]complex128, Width)
			inputs[symbol] = cmplx.Exp(0i)
			for j := 1; j < ComplexOrder; j++ {
				inputs[data[i+j]] = cmplx.Exp(1i * math.Pi * complex(float64(j), 0) / ComplexOrder)
			}
			y := complex128(0)
			for j, value := range inputs {
				y += value * complex128(vector[j])
			}
			y = (y - 1) * (y - 1)
			for j, value := range inputs {
				vector[j] -= complex64(Eta * value * y)
			}
			s[symbols] = vector
		}
		for i, value := range symbols[1:] {
			symbols[i] = value
		}
		symbols[ComplexOrder-1] = symbol
	}
}

// SelfEntropy calculates entropy
func ComplexSelfEntropy(db *bolt.DB, input []byte) (ax []float64) {
	rnd := rand.New(rand.NewSource(1))
	length := len(input)
	weights := NewComplexMatrix(0, Width, length-Order+1)
	orders := make([]int, length-Order+1)
	for i := 0; i < length-Order+1; i++ {
		symbol := Symbols{}
		for j := range symbol {
			symbol[j] = input[i+j]
		}
		var decoded [Width]complex64
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
						r := uint32(output[index])
						index++
						r |= uint32(output[index]) << 8
						index++
						r |= uint32(output[index]) << 16
						index++
						r |= uint32(output[index]) << 24
						index++

						i := uint32(output[index])
						index++
						i |= uint32(output[index]) << 8
						index++
						i |= uint32(output[index]) << 16
						index++
						i |= uint32(output[index]) << 24
						index++
						decoded[key] = complex(math.Float32frombits(r), math.Float32frombits(i))
					}
					return nil
				}
			}
			return nil
		})
		if !found {
			orders[i] = Order - 1
			factor := math.Sqrt(2.0 / float64(Width))
			vector, sum := make([]complex128, Width), complex128(0.0)
			for key := range vector {
				v := complex(rnd.NormFloat64()*factor, rnd.NormFloat64()*factor)
				sum += v * v
				vector[key] = v
			}
			length := cmplx.Sqrt(sum)
			for i, v := range vector {
				vector[i] = v / length
			}
			for _, value := range vector {
				weights.Data = append(weights.Data, complex64(value))
			}
		} else {
			orders[i] = order
			vector, sum := make([]complex128, Width), complex128(0.0)
			for key, value := range decoded {
				v := complex128(value)
				sum += v * v
				vector[key] = v
			}
			length := cmplx.Sqrt(sum)
			for i, v := range vector {
				vector[i] = v / length
			}
			for _, value := range vector {
				weights.Data = append(weights.Data, complex64(value))
			}
		}
	}

	importance := NewComplexMatrix(0, len(orders), 1)
	for _, order := range orders {
		importance.Data = append(importance.Data, complex(1/float32(Order-order), 0))
	}

	entropy := make([]float64, 1)
	entropy[0] = FastComplexSelfEntropyKernel(weights, weights, weights, importance)

	return entropy
}

func markovComplexSelfEntropy() {
	db, err := bolt.Open(*FlagModel, 0600, nil)
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
			entropy := ComplexSelfEntropy(db, n)
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
