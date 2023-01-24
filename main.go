// Copyright 2023 The Lit Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"archive/zip"
	"bufio"
	"compress/gzip"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
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
type Symbols [2]uint8

// SymbolVectors are markov symbol vectors
type SymbolVectors map[Symbols][]float64

// NewSymbolVectors makes new markov symbol vector model
func NewSymbolVectors() SymbolVectors {
	vectors := make(SymbolVectors)
	archive, err := zip.OpenReader("10-0.zip")
	if err != nil {
		panic(err)
	}
	defer archive.Close()
	input, err := archive.File[0].Open()
	if err != nil {
		panic(err)
	}
	defer input.Close()
	data, err := ioutil.ReadAll(input)
	if err != nil {
		panic(err)
	}
	var symbols Symbols
	var prefix uint8
	symbols[1] = data[0]
	symbols[0] = data[1]
	for _, symbol := range data[2:] {
		vector := vectors[symbols]
		if vector == nil {
			vector = make([]float64, 256)
		}
		vector[prefix]++
		vector[symbol]++
		vectors[symbols] = vector
		prefix, symbols[1], symbols[0] = symbols[1], symbols[0], symbol
	}

	for _, vector := range vectors {
		sum := 0.0
		for _, v := range vector {
			sum += v * v
		}
		length := math.Sqrt(sum)
		for i, v := range vector {
			vector[i] = v / length
		}
	}
	return vectors
}

// Entropy calculates entropy
func (v SymbolVectors) Entropy(input []byte) (ax []float64) {
	width := 256
	filler := make([]float64, width)
	length := len(input)
	weights := NewMatrix(width, length-1)
	for i := 1; i < length; i++ {
		symbol := Symbols{input[i], input[i-1]}
		vector := v[symbol]
		if vector == nil {
			vector = filler
		}
		weights.Data = append(weights.Data, vector...)
	}

	l1 := Softmax(Mul(weights, weights))
	l2 := Softmax(Mul(T(weights), l1))
	entropy := Entropy(l2)

	return entropy.Data
}

var (
	// FlagMarkov mode use markov symbol vectors
	FlagMarkov = flag.Bool("markov", false, "markov symbol vector mode")
)

func markov() {
	s := NewSymbolVectors()
	fmt.Println(len(s))
	input := []byte("1:3 And God said, Let there be light: and there was light")
	min, symbol := math.MaxFloat64, 0
	for i := 0; i < 256; i++ {
		entropy := s.Entropy(append(input, byte(i)))
		sum := 0.0
		for _, e := range entropy {
			sum += e
		}
		if sum < min {
			min, symbol = sum, i
		}
	}
	fmt.Printf("%f %c\n", min, symbol)
}

func main() {
	flag.Parse()

	if *FlagMarkov {
		markov()
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
