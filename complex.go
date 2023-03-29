// Copyright 2023 The Lit Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand"
	"path/filepath"
	"runtime"
	"strings"

	zim "github.com/akhenakh/gozim"
	"github.com/k3a/html2text"
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
