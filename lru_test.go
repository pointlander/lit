// Copyright 2023 The Lit Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"testing"
)

func TestLRU(t *testing.T) {
	lru := NewLRU(8)
	for i := 0; i < 12; i++ {
		_, ok := lru.Get(Symbols{uint8(i)})
		if ok {
			t.Fatal("node should be not be found")
		}
		node := lru.Flush()
		if i < 7 && node != nil {
			t.Fatal("no nodes should be flushed", i)
		} else if i == 7 {
			if node == nil {
				t.Fatal("nodes should be flushed", i)
			} else {
				state, j := []uint8{3, 2, 1, 0}, 0
				for node != nil {
					if state[j] != node.Key[0] {
						t.Fatal("state doesn't match")
					}
					j++
					node = node.B
				}
			}
		} else if i > 7 && i < 11 && node != nil {
			t.Fatal("no nodes should be flushed", i)
		} else if i == 11 {
			if node == nil {
				t.Fatal("nodes should be flushed", i)
			} else {
				state, j := []uint8{7, 6, 5, 4}, 0
				for node != nil {
					if state[j] != node.Key[0] {
						t.Fatal("state doesn't match")
					}
					j++
					node = node.B
				}
			}
		}
	}

	check := func(key uint8, state []uint8) {
		if _, ok := lru.Get(Symbols{key}); !ok {
			t.Fatalf("key %d should be found", key)
		}
		node, i := lru.Head, 0
		t.Log("forward", state)
		for node != nil {
			t.Log(node.Key)
			if state[i] != node.Key[0] {
				t.Fatal("invalid key", state[i], node.Key, key)
			}
			i++
			node = node.B
		}

		node, i = lru.Tail, len(state)
		t.Log("backward", state)
		for node != nil {
			i--
			t.Log(node.Key)
			if state[i] != node.Key[0] {
				t.Fatal("invalid key", state[i], node.Key, key)
			}
			node = node.F
		}
	}
	check(8, []uint8{8, 11, 10, 9})
	check(10, []uint8{10, 8, 11, 9})
	check(8, []uint8{8, 10, 11, 9})
	check(8, []uint8{8, 10, 11, 9})

	lru = NewLRU(2)
	if _, ok := lru.Get(Symbols{0}); ok {
		t.Fatal("should not find key 0")
	}
	if lru.Flush() != nil {
		t.Fatal("there shouldn't be a flush")
	}
	if _, ok := lru.Get(Symbols{1}); ok {
		t.Fatal("should not find key 1")
	}
	if lru.Flush() == nil {
		t.Fatal("there should be a flush")
	}
	check(1, []uint8{1})
	check(1, []uint8{1})
	check(1, []uint8{1})
}
