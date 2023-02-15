// Copyright 2020 The Gradient Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64
// +build amd64

package main

import (
	"github.com/ziutek/blas"
)

func dot(X, Y []float64) float64 {
	return blas.Ddot(len(X), X, 1, Y, 1)
}

func axpy(alpha float64, X []float64, Y []float64) {
	blas.Daxpy(len(X), alpha, X, 1, Y, 1)
}
