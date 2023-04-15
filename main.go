// Copyright 2023 The Lit Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"flag"
	"fmt"
	"math"
	"os"
	"sort"

	"github.com/pointlander/compress"
	"github.com/pointlander/pagerank"

	bolt "go.etcd.io/bbolt"
)

const (
	// Order is the order of the markov word vector model
	Order = 9
	// ComplexOrder is the order of the markov word complex vector model
	ComplexOrder = 2
	// Depth is the depth of the search
	Depth = 2
	// Size is the number of histograms
	Size = 1
	// Width is the width of the probability distribution
	Width = Size * 256
)

var (
	// FlagSquare uses square markov model
	FlagSquare = flag.Bool("square", false, "square markov model")
	// FlagMarkov mode uses markov symbol vectors
	FlagMarkov = flag.Bool("markov", false, "markov symbol vector mode")
	// FlagAttention mode uses markov symbols with attention
	FlagAttention = flag.Bool("attention", false, "markov symbol attention mode")
	// FlagDiffusion is a diffusion based model
	FlagDiffusion = flag.Bool("diffusion", false, "diffusion mode")
	// FlagInput is the input into the markov model
	FlagInput = flag.String("input", "What color is the sky?", "input into the markov model")
	// FlagRandomInput use random input
	FlagRandomInput = flag.Int("randomInput", 0, "random string")
	// FlagPagerank page rank mode
	FlagPageRank = flag.Bool("pagerank", false, "pagerank mode")
	// FlagLearn learn a model
	FlagLearn = flag.Bool("learn", false, "learns a model")
	// FlagData is the path to the training data
	FlagData = flag.String("data", "gutenberg_en_all_2022-04.zim", "path to the training data")
	// FlagModel is the model for inference
	FlagModel = flag.String("model", "model.bolt", "the learned model")
	// FlagEntropy calculate the self entropy of a string
	FlagEntropy = flag.String("entropy", "", "calculate the self entropy of a string")
	// FlagRanom select random books from gutenberg for training
	FlagRandom = flag.Bool("random", false, "use random books from gutenberg")
	// FlagScale the scaling factor for the amount of samples
	FlagScale = flag.Int("scale", 8, "the scaling factor for the amount of samples")
	// FlagComplex complex number model
	FlagComplex = flag.Bool("complex", false, "complex model")
)

type Result struct {
	Entropy float64
	Output  []byte
}

func main() {
	flag.Parse()

	if *FlagMarkov {
		markov()
		return
	} else if *FlagAttention && *FlagComplex {
		markovComplexSelfEntropy()
		return
	} else if *FlagAttention {
		markovSelfEntropy()
		return
	} else if *FlagDiffusion {
		markovSelfEntropyDiffusion()
		return
	} else if *FlagPageRank {
		db, err := bolt.Open(*FlagModel, 0600, nil)
		if err != nil {
			panic(err)
		}
		defer db.Close()

		lookup := func(symbol Symbols) (found bool, vector []float64) {
			var decoded [Width]uint16
			db.View(func(tx *bolt.Tx) error {
				b := tx.Bucket([]byte("markov"))
				v := b.Get(symbol[:])
				if v != nil {
					found = true
					index, buffer, output := 0, bytes.NewBuffer(v), make([]byte, 2*Width)
					compress.Mark1Decompress1(buffer, output)
					for key := range decoded {
						decoded[key] = uint16(output[index])
						index++
						decoded[key] |= uint16(output[index]) << 8
						index++
					}
					return nil
				}
				return nil
			})
			if !found {
				return found, nil
			}
			vector, sum := make([]float64, Width), float64(0.0)
			for key, value := range decoded {
				v := float64(value)
				sum += v * v
				vector[key] = v
			}
			length := math.Sqrt(sum)
			for i, v := range vector {
				vector[i] = v / length
			}
			return found, vector
		}

		graph := pagerank.NewGraph64()
		for i := 0; i < Width*Width; i++ {
			x := Symbols{}
			x[Order-2] = byte(i >> 8)
			x[Order-1] = byte(i & 0xff)
			found, a := lookup(x)
			if !found {
				continue
			}
			for j := 0; j < Width*Width; j++ {
				y := Symbols{}
				y[Order-2] = byte(j >> 8)
				y[Order-1] = byte(j & 0xff)
				found, b := lookup(y)
				if !found {
					continue
				}
				sum := 0.0
				for k, value := range a {
					sum += value * b[k]
				}
				graph.Link(uint64(i), uint64(j), sum)
			}
		}
		fmt.Println("graph built")
		type Node struct {
			Node int
			Rank float64
		}
		nodes := make([]Node, 0, 8)
		graph.Rank(0.85, 1e-12, func(node uint64, rank float64) {
			nodes = append(nodes, Node{
				Node: int(node),
				Rank: rank,
			})
		})
		fmt.Println("ranking done")
		sort.Slice(nodes, func(i, j int) bool {
			return nodes[i].Rank > nodes[j].Rank
		})
		fmt.Println("sorting done")
		output, err := os.Create("output.txt")
		if err != nil {
			panic(err)
		}
		defer output.Close()
		for _, node := range nodes {
			fmt.Fprintf(output, "%04x %.12f\n", node.Node, node.Rank)
		}
		return
	} else if *FlagLearn && *FlagComplex {
		var s ComplexSymbolVectors
		if *FlagRandom {
			s = NewComplexSymbolVectorsRandom()
		} else {
			s = NewComplexSymbolVectors()
		}

		fmt.Println("done building")
		db, err := bolt.Open(*FlagModel, 0666, nil)
		if err != nil {
			panic(err)
		}
		defer db.Close()
		db.Update(func(tx *bolt.Tx) error {
			_, err := tx.CreateBucket([]byte("markov"))
			if err != nil {
				panic(err)
			}
			return nil
		})
		fmt.Println("write file")
		type Pair struct {
			Key   []byte
			Value []byte
		}
		length, count, i, pairs := len(s), 0, 0, [1024]Pair{}
		for key, value := range s {
			k := make([]byte, len(key))
			copy(k, key[:])
			pairs[i].Key = k
			index, data := 0, make([]byte, 8*Width)
			for _, v := range value {
				r := math.Float32bits(float32(real(complex128(v))))
				data[index] = byte(r & 0xff)
				index++
				data[index] = byte((r >> 8) & 0xff)
				index++
				data[index] = byte((r >> 16) & 0xff)
				index++
				data[index] = byte((r >> 24) & 0xff)
				index++

				i := math.Float32bits(float32(imag(complex128(v))))
				data[index] = byte(i & 0xff)
				index++
				data[index] = byte((i >> 8) & 0xff)
				index++
				data[index] = byte((i >> 16) & 0xff)
				index++
				data[index] = byte((i >> 24) & 0xff)
				index++
			}
			pairs[i].Value = data
			delete(s, key)
			i++
			count++
			if i == len(pairs) {
				db.Update(func(tx *bolt.Tx) error {
					b := tx.Bucket([]byte("markov"))
					for _, pair := range pairs {
						buffer := bytes.Buffer{}
						compress.Mark1Compress1(pair.Value, &buffer)
						err := b.Put(pair.Key, buffer.Bytes())
						if err != nil {
							return err
						}
					}
					return nil
				})
				i = 0
				fmt.Printf("%f\n", float64(count)/float64(length))
			}
		}
		if i > 0 {
			db.Update(func(tx *bolt.Tx) error {
				b := tx.Bucket([]byte("markov"))
				for _, pair := range pairs[:i] {
					buffer := bytes.Buffer{}
					compress.Mark1Compress1(pair.Value, &buffer)
					err := b.Put(pair.Key, buffer.Bytes())
					if err != nil {
						return err
					}
				}
				return nil
			})
		}
		fmt.Println("done writing file")
		return
	} else if *FlagLearn {
		var s LRU
		if *FlagRandom {
			s = NewSymbolVectorsRandom()
		} else {
			s = NewSymbolVectors()
		}
		s.Close()

		fmt.Println("done building")
		db, err := bolt.Open(*FlagModel, 0666, nil)
		if err != nil {
			panic(err)
		}
		defer db.Close()
		db.Update(func(tx *bolt.Tx) error {
			_, err := tx.CreateBucket([]byte("markov"))
			if err != nil {
				panic(err)
			}
			return nil
		})
		fmt.Println("write file")
		type Pair struct {
			Key   []byte
			Value []byte
		}
		length, count, i, pairs := len(s.Model), 0, 0, [1024]Pair{}
		for key, value := range s.Model {
			k := make([]byte, len(key))
			copy(k, key[:])
			pairs[i].Key = k
			pairs[i].Value = value
			delete(s.Model, key)
			i++
			count++
			if i == len(pairs) {
				db.Update(func(tx *bolt.Tx) error {
					b := tx.Bucket([]byte("markov"))
					for _, pair := range pairs {
						buffer := bytes.Buffer{}
						compress.Mark1Compress1(pair.Value, &buffer)
						err := b.Put(pair.Key, buffer.Bytes())
						if err != nil {
							return err
						}
					}
					return nil
				})
				i = 0
				fmt.Printf("%f\n", float64(count)/float64(length))
			}
		}
		if i > 0 {
			db.Update(func(tx *bolt.Tx) error {
				b := tx.Bucket([]byte("markov"))
				for _, pair := range pairs[:i] {
					buffer := bytes.Buffer{}
					compress.Mark1Compress1(pair.Value, &buffer)
					err := b.Put(pair.Key, buffer.Bytes())
					if err != nil {
						return err
					}
				}
				return nil
			})
		}
		fmt.Println("done writing file")
		return
	} else if *FlagSquare {
		s := NewSquareRandom()
		_ = s
		return
	} else if *FlagEntropy != "" {
		db, err := bolt.Open(*FlagModel, 0600, nil)
		if err != nil {
			panic(err)
		}
		defer db.Close()

		input := []byte(*FlagEntropy)
		entropy := SelfEntropy(db, input, nil)
		fmt.Println(entropy[0] / float64(len(input)))
		return
	}

	v := NewVectors("cc.en.300.vec.gz")
	v.Test()
}
