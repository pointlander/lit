// Copyright 2023 The Lit Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand"

	"github.com/pointlander/pagerank"
)

const (
	// S is the scaling factor for the softmax
	S = 1.0 - 1e-300
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
)

// Matrix is a matrix
type Matrix struct {
	Cols   int
	Rows   int
	Data   []float64
	States [][]float64
}

// NewMatrix creates a new matrix
func NewMatrix(states, cols, rows int) Matrix {
	m := Matrix{
		Cols: cols,
		Rows: rows,
		Data: make([]float64, 0, cols*rows),
	}
	if states > 0 {
		m.States = make([][]float64, states)
		for i := range m.States {
			m.States[i] = make([]float64, cols*rows)
		}
	}
	return m
}

// NewRandMatrix creates a new random matrix
func NewRandMatrix(rnd *rand.Rand, states, cols, rows int) Matrix {
	m := Matrix{
		Cols: cols,
		Rows: rows,
		Data: make([]float64, 0, cols*rows),
	}
	factor := math.Sqrt(2.0 / float64(cols))
	for i := 0; i < cols*rows; i++ {
		m.Data = append(m.Data, rnd.NormFloat64()*factor)
	}
	if states > 0 {
		m.States = make([][]float64, states)
		for i := range m.States {
			m.States[i] = make([]float64, cols*rows)
		}
	}
	return m
}

// Size is the size of the matrix
func (m Matrix) Size() int {
	return m.Cols * m.Rows
}

func softmax(values []float64) {
	max := 0.0
	for _, v := range values {
		if v > max {
			max = v
		}
	}
	s := max * S
	sum := 0.0
	for j, value := range values {
		values[j] = math.Exp(value - s)
		sum += values[j]
	}
	for j, value := range values {
		values[j] = value / sum
	}
}

// SelfEntropyKernel computes the self entropy of Q, K V
func SelfEntropyKernel(Q, K, V, I Matrix) float64 {
	entropies, values, sum := make([]float64, V.Cols), make([]float64, K.Rows), 0.0
	V = T(V)
	for i := 0; i < K.Rows; i++ {
		K := K.Data[i*K.Cols : (i+1)*K.Cols]
		for j := 0; j < Q.Rows; j++ {
			Q := Q.Data[j*Q.Cols : (j+1)*Q.Cols]
			values[j] = dot(K, Q)
		}
		softmax(values)

		for j := 0; j < V.Rows; j++ {
			V := V.Data[j*V.Cols : (j+1)*V.Cols]
			entropies[j] = dot(values, V)
		}
		softmax(entropies)

		entropy := 0.0
		for _, e := range entropies {
			entropy += e * math.Log(e)
		}
		sum -= entropy * I.Data[i]
	}
	return sum
}

// DirectSelfEntropyKernel computes the self entropy of Q, K, V
func DirectSelfEntropyKernel(Q, K, V, I Matrix) []float64 {
	entropies, values, results := make([]float64, V.Cols), make([]float64, K.Rows), make([]float64, 0, K.Rows)
	V = T(V)
	for i := 0; i < K.Rows; i++ {
		K := K.Data[i*K.Cols : (i+1)*K.Cols]
		for j := 0; j < Q.Rows; j++ {
			Q := Q.Data[j*Q.Cols : (j+1)*Q.Cols]
			values[j] = dot(K, Q)
		}
		softmax(values)

		for j := 0; j < V.Rows; j++ {
			V := V.Data[j*V.Cols : (j+1)*V.Cols]
			entropies[j] = dot(values, V)
		}
		softmax(entropies)

		entropy := 0.0
		for _, e := range entropies {
			entropy += e * math.Log(e)
		}
		results = append(results, entropy)
	}
	if len(I.Data) > 0 {
		for key, value := range results {
			results[key] = value * I.Data[key]
		}
	}
	return results
}

// https://arxiv.org/abs/1511.05042
func spherical(values []float64) {
	sum := 0.0
	for j, value := range values {
		values[j] = value*value/2 + value + 1
		sum += values[j]
	}
	for j, value := range values {
		values[j] = value / sum
	}
}

// FastSelfEntropyKernel computes the fast self entropy of Q, K V
func FastSelfEntropyKernel(Q, K, V, I Matrix) float64 {
	entropies, values, sum := make([]float64, V.Cols), make([]float64, K.Rows), 0.0
	V = T(V)
	for i := 0; i < K.Rows; i++ {
		K := K.Data[i*K.Cols : (i+1)*K.Cols]
		for j := 0; j < Q.Rows; j++ {
			Q := Q.Data[j*Q.Cols : (j+1)*Q.Cols]
			values[j] = dot(K, Q)
		}
		spherical(values)

		for j := 0; j < V.Rows; j++ {
			V := V.Data[j*V.Cols : (j+1)*V.Cols]
			entropies[j] = dot(values, V)
		}
		spherical(entropies)

		entropy := 0.0
		for _, e := range entropies {
			entropy += e * math.Log2(e)
		}
		sum -= entropy * I.Data[i]
	}
	return sum
}

// Mul multiplies two matrices
func Mul(m Matrix, n Matrix) Matrix {
	if m.Cols != n.Cols {
		panic(fmt.Errorf("%d != %d", m.Cols, n.Cols))
	}
	columns := m.Cols
	o := Matrix{
		Cols: m.Rows,
		Rows: n.Rows,
		Data: make([]float64, 0, m.Rows*n.Rows),
	}
	lenn, lenm := len(n.Data), len(m.Data)
	for i := 0; i < lenn; i += columns {
		nn := n.Data[i : i+columns]
		for j := 0; j < lenm; j += columns {
			mm := m.Data[j : j+columns]
			o.Data = append(o.Data, dot(mm, nn))
		}
	}
	return o
}

// H element wise multiplies two matrices
func H(m Matrix, n Matrix) Matrix {
	lena, lenb := len(m.Data), len(n.Data)
	if lena%lenb != 0 {
		panic(fmt.Errorf("%d %% %d != 0", lena, lenb))
	}

	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float64, 0, m.Cols*m.Rows),
	}
	for i, value := range m.Data {
		o.Data = append(o.Data, value*n.Data[i%lenb])
	}
	return o
}

// Add adds two matrices
func Add(m Matrix, n Matrix) Matrix {
	lena, lenb := len(m.Data), len(n.Data)
	if lena%lenb != 0 {
		panic(fmt.Errorf("%d %% %d != 0", lena, lenb))
	}

	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float64, 0, m.Cols*m.Rows),
	}
	for i, value := range m.Data {
		o.Data = append(o.Data, value+n.Data[i%lenb])
	}
	return o
}

// Sub subtracts two matrices
func Sub(m Matrix, n Matrix) Matrix {
	lena, lenb := len(m.Data), len(n.Data)
	if lena%lenb != 0 {
		panic(fmt.Errorf("%d %% %d != 0", lena, lenb))
	}

	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float64, 0, m.Cols*m.Rows),
	}
	for i, value := range m.Data {
		o.Data = append(o.Data, value-n.Data[i%lenb])
	}
	return o
}

// Softmax is the softmax of a matrix
func Softmax(m Matrix) Matrix {
	size, width := len(m.Data), m.Cols
	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float64, 0, m.Cols*m.Rows),
	}
	max := 0.0
	for _, v := range m.Data {
		if v > max {
			max = v
		}
	}
	values := make([]float64, width)
	for i := 0; i < size; i += width {
		s := max * S
		sum := 0.0
		for j, ax := range m.Data[i : i+width] {
			values[j] = math.Exp(ax - s)
			sum += values[j]
		}
		for _, cx := range values {
			o.Data = append(o.Data, cx/sum)
		}
	}
	return o
}

// Normalize normalizes a matrix to the unit vector
func Normalize(m Matrix) Matrix {
	size, width := len(m.Data), m.Cols
	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float64, 0, m.Cols*m.Rows),
	}
	for i := 0; i < size; i += width {
		sum := 0.0
		for _, ax := range m.Data[i : i+width] {
			sum += ax * ax
		}
		length := math.Sqrt(sum)
		if sum == 0 {
			length = 1
		}
		for _, ax := range m.Data[i : i+width] {
			o.Data = append(o.Data, ax/length)
		}
	}
	return o
}

// Entropy is the entropy of the matrix
func Entropy(m Matrix) Matrix {
	size, width := len(m.Data), m.Cols
	o := Matrix{
		Cols: m.Rows,
		Rows: 1,
		Data: make([]float64, 0, m.Rows),
	}
	for i := 0; i < size; i += width {
		sum := 0.0
		for k := 0; k < width; k++ {
			ax := m.Data[i+k]
			sum += ax * math.Log(ax)
		}
		o.Data = append(o.Data, -sum)
	}
	return o
}

// Neg negates a matrix
func Neg(m Matrix) Matrix {
	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float64, 0, m.Cols*m.Rows),
	}
	for _, value := range m.Data {
		o.Data = append(o.Data, -value)
	}
	return o
}

// Logis computes the logis of a matrix
func Logis(m Matrix) Matrix {
	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float64, 0, m.Cols*m.Rows),
	}
	for _, value := range m.Data {
		o.Data = append(o.Data, 1/(1+math.Exp(-value)))
	}
	return o
}

func logis(value float64) float64 {
	return 1 / (1 + math.Exp(-value))
}

// DLogis computes the dlogis of a matrix
func DLogis(m Matrix) Matrix {
	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float64, 0, m.Cols*m.Rows),
	}
	for _, value := range m.Data {
		o.Data = append(o.Data, logis(value)*(1-logis(value)))
	}
	return o
}

// T tramsposes a matrix
func T(m Matrix) Matrix {
	o := Matrix{
		Cols: m.Rows,
		Rows: m.Cols,
		Data: make([]float64, 0, m.Cols*m.Rows),
	}
	for i := 0; i < m.Cols; i++ {
		for j := 0; j < m.Rows; j++ {
			o.Data = append(o.Data, m.Data[j*m.Cols+i])
		}
	}
	return o
}

// AppendOne appends 1 to each row of a matrix
func AppendOne(m Matrix) Matrix {
	o := Matrix{
		Cols: m.Cols + 1,
		Rows: m.Rows,
		Data: make([]float64, 0, (m.Cols+1)*m.Rows),
	}
	length := len(m.Data)
	for i := 0; i < length; i += m.Cols {
		o.Data = append(o.Data, m.Data[i:i+m.Cols]...)
		o.Data = append(o.Data, 1.0)
	}
	return o
}

// PageRank computes the page rank of the adjacency matrix
func PageRank(m Matrix) Matrix {
	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float64, m.Cols),
	}
	graph := pagerank.NewGraph64()
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			graph.Link(uint64(i), uint64(j), m.Data[i*m.Cols+j])
		}
	}
	graph.Rank(0.85, 0.000001, func(node uint64, rank float64) {
		o.Data[node] = rank
	})
	return o
}

// ComplexMatrix is a complex matrix
type ComplexMatrix struct {
	Cols   int
	Rows   int
	Data   []complex64
	States [][]complex64
}

// NewComplexMatrix creates a new complex matrix
func NewComplexMatrix(states, cols, rows int) ComplexMatrix {
	m := ComplexMatrix{
		Cols: cols,
		Rows: rows,
		Data: make([]complex64, 0, cols*rows),
	}
	if states > 0 {
		m.States = make([][]complex64, states)
		for i := range m.States {
			m.States[i] = make([]complex64, cols*rows)
		}
	}
	return m
}

// NewRandComplexMatrix creates a new random matrix
func NewRandComplexMatrix(rnd *rand.Rand, states, cols, rows int) ComplexMatrix {
	m := ComplexMatrix{
		Cols: cols,
		Rows: rows,
		Data: make([]complex64, 0, cols*rows),
	}
	factor := math.Sqrt(2.0 / float64(cols))
	for i := 0; i < cols*rows; i++ {
		m.Data = append(m.Data, complex(float32(rnd.NormFloat64()*factor), float32(rnd.NormFloat64()*factor)))
	}
	if states > 0 {
		m.States = make([][]complex64, states)
		for i := range m.States {
			m.States[i] = make([]complex64, cols*rows)
		}
	}
	return m
}

// Size is the size of the matrix
func (m ComplexMatrix) Size() int {
	return m.Cols * m.Rows
}

// https://arxiv.org/abs/1511.05042
func complexSpherical(values []complex64) {
	sum := complex64(0.0)
	for j, value := range values {
		values[j] = value*value/2 + value + 1
		sum += values[j]
	}
	for j, value := range values {
		values[j] = value / sum
	}
}

// FastComplexSelfEntropyKernel computes the fast complex self entropy of Q, K V
func FastComplexSelfEntropyKernel(Q, K, V, I ComplexMatrix) float64 {
	entropies, values, sum := make([]complex64, V.Cols), make([]complex64, K.Rows), complex64(0.0)
	V = ComplexT(V)
	for i := 0; i < K.Rows; i++ {
		K := K.Data[i*K.Cols : (i+1)*K.Cols]
		for j := 0; j < Q.Rows; j++ {
			Q := Q.Data[j*Q.Cols : (j+1)*Q.Cols]
			for k, value := range K {
				values[j] += value * Q[k]
			}
		}
		complexSpherical(values)

		for j := 0; j < V.Rows; j++ {
			V := V.Data[j*V.Cols : (j+1)*V.Cols]
			for k, value := range values {
				entropies[j] += value * V[k]
			}
		}
		complexSpherical(entropies)

		entropy := complex64(0.0)
		for _, e := range entropies {
			entropy += e * complex64(cmplx.Log(complex128(e)))
		}
		sum -= entropy * I.Data[i]
	}
	return cmplx.Abs(complex128(sum))
}

// ComplexMul multiplies two complex matrices
func ComplexMul(m ComplexMatrix, n ComplexMatrix) ComplexMatrix {
	if m.Cols != n.Cols {
		panic(fmt.Errorf("%d != %d", m.Cols, n.Cols))
	}
	columns := m.Cols
	o := ComplexMatrix{
		Cols: m.Rows,
		Rows: n.Rows,
		Data: make([]complex64, 0, m.Rows*n.Rows),
	}
	lenn, lenm := len(n.Data), len(m.Data)
	for i := 0; i < lenn; i += columns {
		nn := n.Data[i : i+columns]
		for j := 0; j < lenm; j += columns {
			mm, sum := m.Data[j:j+columns], complex64(0.0)
			for k, value := range mm {
				sum += value * nn[k]
			}
			o.Data = append(o.Data, sum)
		}
	}
	return o
}

// ComplexH element wise multiplies two complex matrices
func ComplexH(m ComplexMatrix, n ComplexMatrix) ComplexMatrix {
	lena, lenb := len(m.Data), len(n.Data)
	if lena%lenb != 0 {
		panic(fmt.Errorf("%d %% %d != 0", lena, lenb))
	}

	o := ComplexMatrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]complex64, 0, m.Cols*m.Rows),
	}
	for i, value := range m.Data {
		o.Data = append(o.Data, value*n.Data[i%lenb])
	}
	return o
}

// ComplexAdd adds two complex matrices
func ComplexAdd(m ComplexMatrix, n ComplexMatrix) ComplexMatrix {
	lena, lenb := len(m.Data), len(n.Data)
	if lena%lenb != 0 {
		panic(fmt.Errorf("%d %% %d != 0", lena, lenb))
	}

	o := ComplexMatrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]complex64, 0, m.Cols*m.Rows),
	}
	for i, value := range m.Data {
		o.Data = append(o.Data, value+n.Data[i%lenb])
	}
	return o
}

// ComplexSub subtracts two complex matrices
func ComplexSub(m ComplexMatrix, n ComplexMatrix) ComplexMatrix {
	lena, lenb := len(m.Data), len(n.Data)
	if lena%lenb != 0 {
		panic(fmt.Errorf("%d %% %d != 0", lena, lenb))
	}

	o := ComplexMatrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]complex64, 0, m.Cols*m.Rows),
	}
	for i, value := range m.Data {
		o.Data = append(o.Data, value-n.Data[i%lenb])
	}
	return o
}

// ComplexSphericalSoftmax is the spherical softmax of a complex matrix
func ComplexSphericalSoftmax(m ComplexMatrix) ComplexMatrix {
	const E = complex(0, 0)
	size, width := len(m.Data), m.Cols
	o := ComplexMatrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]complex64, 0, m.Cols*m.Rows),
	}
	values := make([]complex64, width)
	for i := 0; i < size; i += width {
		sum := complex(float32(0), float32(0))
		for j, ax := range m.Data[i : i+width] {
			values[j] = ax*ax + E
			sum += values[j]
		}
		for _, value := range values {
			o.Data = append(o.Data, value/sum)
		}
	}
	return o
}

// ComplexNormalize normalizes a complex matrix to the unit vector
func ComplexNormalize(m ComplexMatrix) ComplexMatrix {
	size, width := len(m.Data), m.Cols
	o := ComplexMatrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]complex64, 0, m.Cols*m.Rows),
	}
	for i := 0; i < size; i += width {
		sum := complex64(0.0)
		for _, ax := range m.Data[i : i+width] {
			sum += ax * ax
		}
		length := complex64(cmplx.Sqrt(complex128(sum)))
		if sum == 0 {
			length = 1
		}
		for _, ax := range m.Data[i : i+width] {
			o.Data = append(o.Data, ax/length)
		}
	}
	return o
}

// ComplexEntropy is the entropy of the complex matrix
func ComplexEntropy(m ComplexMatrix) ComplexMatrix {
	size, width := len(m.Data), m.Cols
	o := ComplexMatrix{
		Cols: m.Rows,
		Rows: 1,
		Data: make([]complex64, 0, m.Rows),
	}
	for i := 0; i < size; i += width {
		sum := complex64(0.0)
		for k := 0; k < width; k++ {
			ax := m.Data[i+k]
			sum += ax * complex64(cmplx.Log(complex128(ax)))
		}
		o.Data = append(o.Data, -sum)
	}
	return o
}

// ComplexNeg negates a matrix
func ComplexNeg(m ComplexMatrix) ComplexMatrix {
	o := ComplexMatrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]complex64, 0, m.Cols*m.Rows),
	}
	for _, value := range m.Data {
		o.Data = append(o.Data, -value)
	}
	return o
}

// ComplexLogis computes the logis of a complex matrix
func ComplexLogis(m ComplexMatrix) ComplexMatrix {
	o := ComplexMatrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]complex64, 0, m.Cols*m.Rows),
	}
	for _, value := range m.Data {
		o.Data = append(o.Data, 1/(1+complex64(cmplx.Exp(complex128(-value)))))
	}
	return o
}

func complexLogis(value complex64) complex64 {
	return 1 / (1 + complex64(cmplx.Exp(complex128(-value))))
}

// ComplexDLogis computes the dlogis of a complex matrix
func ComplexDLogis(m ComplexMatrix) ComplexMatrix {
	o := ComplexMatrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]complex64, 0, m.Cols*m.Rows),
	}
	for _, value := range m.Data {
		o.Data = append(o.Data, complexLogis(value)*(1-complexLogis(value)))
	}
	return o
}

// ComplexT tramsposes a complex matrix
func ComplexT(m ComplexMatrix) ComplexMatrix {
	o := ComplexMatrix{
		Cols: m.Rows,
		Rows: m.Cols,
		Data: make([]complex64, 0, m.Cols*m.Rows),
	}
	for i := 0; i < m.Cols; i++ {
		for j := 0; j < m.Rows; j++ {
			o.Data = append(o.Data, m.Data[j*m.Cols+i])
		}
	}
	return o
}

// ComplexAppendOne appends 1 to each row of a complex matrix
func ComplexAppendOne(m ComplexMatrix) ComplexMatrix {
	o := ComplexMatrix{
		Cols: m.Cols + 1,
		Rows: m.Rows,
		Data: make([]complex64, 0, (m.Cols+1)*m.Rows),
	}
	length := len(m.Data)
	for i := 0; i < length; i += m.Cols {
		o.Data = append(o.Data, m.Data[i:i+m.Cols]...)
		o.Data = append(o.Data, 1.0)
	}
	return o
}
