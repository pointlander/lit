// Copyright 2023 The Lit Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"

	"github.com/pointlander/compress"
)

// Node is an entry in the LRU cache
type Node struct {
	F, B  *Node
	Value []uint16
	Key   Symbols
	Seen  bool
}

// LRU is a least recently used cache
type LRU struct {
	Size       int
	Head, Tail *Node
	Nodes      map[Symbols]*Node
	Model      map[Symbols][]uint8
}

// NewLRU creates a new LRU cache
func NewLRU(size uint) LRU {
	if size == 0 {
		panic("size should not be 0")
	}
	return LRU{
		Size:  1 << size,
		Model: make(map[Symbols][]uint8),
	}
}

// Flush flush the oldest entries in the cache
func (l *LRU) Flush() *Node {
	size := l.Size
	if len(l.Nodes) < size {
		return nil
	}

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
	write()
	size >>= 1
	for i := 1; i < size; i++ {
		node = node.F
		write()
	}
	node.F.B, l.Tail, node.F = nil, node.F, nil

	return node
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
