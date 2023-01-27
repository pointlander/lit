// Copyright 2023 The Lit Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"archive/zip"
	"bufio"
	"compress/gzip"
	"encoding/gob"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"math"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"

	zim "github.com/akhenakh/gozim"
	"github.com/k3a/html2text"
)

const (
	// Order is the order of the markov word vector model
	Order = 4
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
type SymbolVectors map[Symbols]*[256]float32

// NewSymbolVectors makes new markov symbol vector model
func NewSymbolVectors() SymbolVectors {
	vectors := make(SymbolVectors)
	learn := func(data []byte) {
		var symbols Symbols
		var prefix uint8
		for i, symbol := range data[:len(data)-Order+1] {
			vector := vectors[symbols]
			if vector == nil {
				vector = &[256]float32{}
			}
			//vector[prefix]++
			_ = prefix
			vector[symbol]++
			for j := 1; j < Order; j++ {
				vector[data[i+j]]++
			}
			vectors[symbols] = vector
			prefix = symbols[0]
			for i, value := range symbols[1:] {
				symbols[i] = value
			}
			symbols[Order-1] = symbol
		}
	}
	load := func(book string) {
		archive, err := zip.OpenReader(book)
		if err != nil {
			panic(err)
		}
		defer archive.Close()
		fmt.Println("open book", archive.File[0].Name)
		input, err := archive.File[0].Open()
		if err != nil {
			panic(err)
		}
		defer input.Close()
		data, err := ioutil.ReadAll(input)
		if err != nil {
			panic(err)
		}
		learn(data)
	}
	reader, err := zim.NewReader("/home/andrew/Downloads/gutenberg_en_all_2022-04.zim", false)
	if err != nil {
		panic(err)
	}
	var m runtime.MemStats
	articles := reader.ListArticles()
	for article := range articles {
		url := article.FullURL()
		if strings.HasSuffix(url, ".html") {
			html, err := article.Data()
			if err != nil {
				panic(err)
			}
			plain := html2text.HTML2Text(string(html))
			runtime.ReadMemStats(&m)
			fmt.Println(url, "Alloc", m.Alloc/(1024*1024))
			learn([]byte(plain))
		}
	}
	load("books/10-0.zip")
	load("books/100-0.zip")
	load("books/145-0.zip")
	load("books/1513-0.zip")
	load("books/16389-0.zip")
	load("books/2641-0.zip")
	load("books/2701-0.zip")
	load("books/37106.zip")
	load("books/394-0.zip")
	load("books/67979-0.zip")
	for _, vector := range vectors {
		sum := float32(0.0)
		for _, v := range vector {
			sum += v * v
		}
		length := float32(math.Sqrt(float64(sum)))
		for i, v := range vector {
			vector[i] = v / length
		}
	}
	return vectors
}

// Entropy calculates entropy
func (v SymbolVectors) Entropy(input []byte) (ax []float64) {
	width := 256
	filler := [256]float32{}
	length := len(input)
	weights := NewMatrix(width, length-Order+1)
	for i := 0; i < length-Order+1; i++ {
		symbol := Symbols{}
		for j := range symbol {
			symbol[j] = input[i+j]
		}
		vector := v[symbol]
		if vector == nil {
			vector = &filler
		}
		for _, value := range vector {
			weights.Data = append(weights.Data, float64(value))
		}
	}

	l1 := Softmax(Mul(weights, weights))
	l2 := Softmax(Mul(T(weights), l1))
	entropy := Entropy(l2)

	return entropy.Data
}

var (
	// FlagMarkov mode use markov symbol vectors
	FlagMarkov = flag.Bool("markov", false, "markov symbol vector mode")
	// FlagLearn learn a model
	FlagLearn = flag.Bool("learn", false, "learns a model")
)

func markov() {
	var s SymbolVectors
	input, err := os.Open("model.bin")
	if err != nil {
		panic(err)
	}
	defer input.Close()
	decompressor, err := gzip.NewReader(input)
	if err != nil {
		panic(err)
	}
	defer decompressor.Close()
	decoder := gob.NewDecoder(decompressor)
	err = decoder.Decode(&s)
	if err != nil {
		panic(err)
	}

	fmt.Println(float64(len(s)) / math.Pow(float64(256), Order))
	//in := []byte("1:3 And God said, Let there be light: and there was light")
	in := []byte("What color is the sky?")
	var search func(depth int, input []byte) (entropy float64, output []byte)
	search = func(depth int, input []byte) (entropy float64, output []byte) {
		if depth == 0 {
			ent := s.Entropy(input)
			for _, value := range ent {
				entropy += value
			}
			return entropy, input
		}
		min, s := math.MaxFloat64, []byte{}
		for i := 0; i < 256; i++ {
			n := make([]byte, len(input))
			copy(n, input)
			e, o := search(depth-1, append(n, byte(i)))
			if e < min {
				min, s = e, o
			}
		}
		return min, s
	}
	entropy, output := search(2, in)
	fmt.Println(entropy, string(output))
	fmt.Printf("\n")
	for i := 0; i < 128; i++ {
		entropy, output = search(2, output)
		fmt.Println(entropy, string(output))
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
		s := NewSymbolVectors()
		output, err := os.Create("model.bin")
		if err != nil {
			panic(err)
		}
		compressor := gzip.NewWriter(output)
		defer compressor.Close()
		encoder := gob.NewEncoder(compressor)
		err = encoder.Encode(s)
		if err != nil {
			panic(err)
		}
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
