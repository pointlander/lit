// Copyright 2023 The Lit Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math/rand"
	"testing"
)

// Length is the length of the matrix
const Length = 128

func BenchmarkSelfEntropy(b *testing.B) {
	rnd := rand.New(rand.NewSource(1))
	weights, importance := NewRandMatrix(rnd, 0, Width, Length), NewRandMatrix(rnd, 0, Length, 1)
	for n := 0; n < b.N; n++ {
		l1 := Softmax(Mul(weights, weights))
		l2 := Softmax(Mul(T(weights), l1))
		entropy := H(Entropy(l2), importance)
		sum := 0.0
		for _, value := range entropy.Data {
			sum += value
		}
	}
}

func BenchmarkSelfEntropyKernel(b *testing.B) {
	rnd := rand.New(rand.NewSource(1))
	weights, importance := NewRandMatrix(rnd, 0, Width, Length), NewRandMatrix(rnd, 0, Length, 1)
	for n := 0; n < b.N; n++ {
		SelfEntropyKernel(weights, weights, weights, importance)
	}
}

func BenchmarkFastSelfEntropyKernel(b *testing.B) {
	rnd := rand.New(rand.NewSource(1))
	weights, importance := NewRandMatrix(rnd, 0, Width, Length), NewRandMatrix(rnd, 0, Length, 1)
	for n := 0; n < b.N; n++ {
		FastSelfEntropyKernel(weights, weights, weights, importance)
	}
}
