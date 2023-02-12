// Copyright 2023 The Lit Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math/rand"
	"testing"
)

func BenchmarkSelfEntropyKernel(b *testing.B) {
	rnd := rand.New(rand.NewSource(1))
	weights, importance := NewRandMatrix(rnd, Width, 128), NewRandMatrix(rnd, 128, 1)
	for n := 0; n < b.N; n++ {
		SelfEntropyKernel(weights, weights, weights, importance)
	}
}

func BenchmarkFastSelfEntropyKernel(b *testing.B) {
	rnd := rand.New(rand.NewSource(1))
	weights, importance := NewRandMatrix(rnd, Width, 128), NewRandMatrix(rnd, 128, 1)
	for n := 0; n < b.N; n++ {
		FastSelfEntropyKernel(weights, weights, weights, importance)
	}
}
