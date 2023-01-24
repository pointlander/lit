// Copyright 2023 The Lit Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bufio"
	"compress/gzip"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/pointlander/gradient/tf32"
)

// Vector is a word vector
type Vector struct {
	Word   string
	Vector []float32
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
		values := make([]float32, 0, len(parts)-1)
		for _, v := range parts[1:] {
			n, err := strconv.ParseFloat(strings.TrimSpace(v), 64)
			if err != nil {
				panic(err)
			}
			values = append(values, float32(n))
		}
		sum := float32(0)
		for _, v := range values {
			sum += v * v
		}
		length := float32(math.Sqrt(float64(sum)))
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
func (v *Vectors) Entropy(input []string) (ax []float32) {
	width := len(v.Dictionary["dog"].Vector)
	length := len(input)
	set := tf32.NewSet()
	set.Add("weights", width, length)
	weights := set.ByName["weights"]
	for _, word := range input {
		vector := v.Dictionary[word].Vector
		weights.X = append(weights.X, vector...)
	}

	l1 := tf32.Softmax(tf32.Mul(set.Get("weights"), set.Get("weights")))
	l2 := tf32.Softmax(tf32.Mul(tf32.T(set.Get("weights")), l1))
	entropy := tf32.Entropy(l2)

	entropy(func(a *tf32.V) bool {
		ax = a.X
		return true
	})
	return
}

func main() {
	v := NewVectors("cc.en.300.vec.gz")

	input := []string{"and", "god", "said", "let", "there", "be", "light", "and", "there", "was"}
	type Word struct {
		Word    string
		Entropy float32
	}
	list := make([]Word, 0, 8)
	target := make([]float32, 300)
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

	words, words2, max, m := []string{}, []string{}, float32(0.0), float32(0.0)
	for _, w := range v.List {
		ab, aa, bb := float32(0.0), float32(0.0), float32(0.0)
		for i, a := range w.Vector {
			b := target[i]
			ab += a * b
			aa += a * a
			bb += b * b
		}
		s := ab / (float32(math.Sqrt(float64(aa))) * float32(math.Sqrt(float64(bb))))
		if s > max {
			max, words = s, append(words, w.Word)
		}

		entropy := v.Entropy(append(input, w.Word))
		/*sum := float32(0.0)
		for _, e := range entropy {
			sum += e
		}*/
		e := entropy[len(entropy)-1]
		if e > m {
			m, words2 = e, append(words2, w.Word)
		}
	}
	fmt.Println(words)
	fmt.Println(words2)
}
