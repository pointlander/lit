// Copyright 2023 The Lit Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"runtime"

	"github.com/pointlander/compress"
)

// Node is an entry in the LRU cache
type Node struct {
	F, B  *Node
	Value []uint16
	Key   Symbols
}

// LRU is a least recently used cache
type LRU struct {
	Size       int
	Head, Tail *Node
	Nodes      map[Symbols]*Node
	Model      map[Symbols][]uint8
}

// NewLRU creates a new LRU cache
func NewLRU(size int) LRU {
	if size == 0 {
		panic("size should not be 0")
	}
	return LRU{
		Size:  size,
		Model: make(map[Symbols][]uint8),
	}
}

// Flush flush the oldest entries in the cache
func (l *LRU) Flush() *Node {
	size := l.Size
	if len(l.Nodes) < size {
		return nil
	}

	type N struct {
		Key   Symbols
		Value []byte
	}
	done := make(chan N, runtime.NumCPU())
	write := func(node *Node) {
		index, data := 0, make([]byte, 2*Width)
		for _, value := range node.Value {
			data[index] = byte(value & 0xff)
			index++
			data[index] = byte((value >> 8) & 0xff)
			index++
		}
		buffer := bytes.Buffer{}
		compress.Mark1Compress1(data, &buffer)
		done <- N{
			Key:   node.Key,
			Value: buffer.Bytes(),
		}
	}
	node := l.Tail
	delete(l.Nodes, node.Key)
	go write(node)
	size >>= 1
	n := <-done
	l.Model[n.Key] = n.Value
	i, j := 1, 0
	for i < size && j < runtime.NumCPU() {
		node = node.F
		delete(l.Nodes, node.Key)
		go write(node)
		i++
		j++
	}
	for i < size {
		n := <-done
		l.Model[n.Key] = n.Value
		j--
		node = node.F
		delete(l.Nodes, node.Key)
		go write(node)
		i++
		j++
	}
	for k := 0; k < j; k++ {
		n := <-done
		l.Model[n.Key] = n.Value
	}
	node.F.B, l.Tail, node.F = nil, node.F, nil
	return node
}

// Flush flush the oldest entries in the cache
func (l *LRU) Close() {
	node := l.Tail
	write := func() {
		delete(l.Nodes, node.Key)
		index, data := 0, make([]byte, 2*Width)
		for _, value := range node.Value {
			data[index] = byte(value & 0xff)
			index++
			data[index] = byte((value >> 8) & 0xff)
			index++
		}
		buffer := bytes.Buffer{}
		compress.Mark1Compress1(data, &buffer)
		l.Model[node.Key] = buffer.Bytes()
	}
	for node != nil {
		write()
		node = node.F
	}
}

// Get gets an entry and sets it as the most recent
func (l *LRU) Get(key Symbols) (*Node, bool) {
	length := len(l.Nodes)
	if length > 0 {
		if node := l.Nodes[key]; node != nil {
			if node.F != nil {
				if node.B != nil {
					node.B.F, node.F.B = node.F, node.B
				} else {
					node.F.B, l.Tail = nil, node.F
				}
				node.F, node.B, l.Head, l.Head.F = nil, l.Head, node, node
			}

			return node, true
		}
	}

	node, compressed := &Node{Key: key}, l.Model[key]
	if compressed != nil {
		decoded, index, buffer, output := make([]uint16, Width), 0, bytes.NewBuffer(compressed), make([]byte, 2*Width)
		compress.Mark1Decompress1(buffer, output)
		for key := range decoded {
			decoded[key] = uint16(output[index])
			index++
			decoded[key] |= uint16(output[index]) << 8
			index++
		}
		node.Value = decoded
	} else {
		node.Value = make([]uint16, Width)
	}
	node.B, l.Head = l.Head, node
	if length == 0 {
		l.Tail = node
		l.Nodes = make(map[Symbols]*Node, l.Size)
	} else {
		node.B.F = node
	}
	l.Nodes[key] = node
	return node, false
}
