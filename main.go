// Copyright 2023 The Lit Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/pointlander/compress"
	"github.com/pointlander/gradient/tf32"
	"github.com/pointlander/pagerank"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"

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

// Indexes are the context indexes for the markov model
var Indexes = [Order]int{0, 1, 2, 3, 4, 5, 6, 7, 8}

//var Indexes = [5]int{0, 3, 5, 7, 8}

var (
	// FlagSquare uses square markov model
	FlagSquare = flag.Bool("square", false, "square markov model")
	// FlagMarkov mode uses markov symbol vectors
	FlagMarkov = flag.Bool("markov", false, "markov symbol vector mode")
	// FlagAttention mode uses markov symbols with attention
	FlagAttention = flag.Bool("attention", false, "markov symbol attention mode")
	// FlagMutual mutal self entropy
	FlagMutual = flag.Bool("mutual", false, "mutual self entropy")
	// FlagMeta mode uses attention of attention
	FlagMeta = flag.Bool("meta", false, "attention of attention")
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
	Symbols []float64
	Output  []byte
}

const (
	// B1 exponential decay of the rate for the first moment estimates
	B1 = 0.9
	// B2 exponential decay rate for the second-moment estimates
	B2 = 0.999
	// Eta is the learning rate
	Eta = .001
)

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
	} else if *FlagMutual {
		markovMutualSelfEntropy()
	} else if *FlagMeta {
		markovDirectSelfEntropy()
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
			x[len(Indexes)-2] = byte(i >> 8)
			x[len(Indexes)-1] = byte(i & 0xff)
			found, a := lookup(x)
			if !found {
				continue
			}
			for j := 0; j < Width*Width; j++ {
				y := Symbols{}
				y[len(Indexes)-2] = byte(j >> 8)
				y[len(Indexes)-1] = byte(j & 0xff)
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
						err := b.Put(pair.Key, pair.Value)
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
					err := b.Put(pair.Key, pair.Value)
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
		s.markovSelfEntropy()
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

	db, err := bolt.Open(*FlagModel, 0600, nil)
	if err != nil {
		panic(err)
	}
	defer db.Close()

	data, err := ioutil.ReadFile("train-v2.0.json")
	if err != nil {
		panic(err)
	}
	var squad Squad
	err = json.Unmarshal(data, &squad)
	if err != nil {
		panic(err)
	}
	type TrainingPair struct {
		Question []byte
		Answer   []byte
	}
	training := []TrainingPair{}
	for _, data := range squad.Data {
		for _, paragraph := range data.Paragraphs {
			for _, question := range paragraph.Qas {
				if question.IsImpossible {
					continue
				}
				training = append(training, TrainingPair{
					Question: []byte(question.Question),
					Answer:   []byte(question.Answers[0].Text),
				})
			}
		}
	}

	rnd := rand.New(rand.NewSource(1))

	set := tf32.NewSet()
	set.Add("w1", 256, 1024)
	set.Add("b1", 1024, 1)
	set.Add("w2", 2048, 256)
	set.Add("b2", 256, 1)
	for _, w := range set.Weights {
		if strings.HasPrefix(w.N, "b") {
			w.X = w.X[:cap(w.X)]
			w.States = make([][]float32, StateTotal)
			for i := range w.States {
				w.States[i] = make([]float32, len(w.X))
			}
			continue
		}
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for i := 0; i < cap(w.X); i++ {
			w.X = append(w.X, float32((2*rnd.Float64()-1)*factor))
		}
		w.States = make([][]float32, StateTotal)
		for i := range w.States {
			w.States[i] = make([]float32, len(w.X))
		}
	}

	others := tf32.NewSet()
	others.Add("inputs", 256, 100)
	others.Add("outputs", 256, 100)
	inputs := others.ByName["inputs"]
	inputs.X = inputs.X[:cap(inputs.X)]
	outputs := others.ByName["outputs"]
	outputs.X = outputs.X[:cap(outputs.X)]

	l1 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("w1"), others.Get("inputs")), set.Get("b1")))
	l2 := tf32.Add(tf32.Mul(set.Get("w2"), l1), set.Get("b2"))
	cost := tf32.Sum(tf32.Quadratic(others.Get("outputs"), l2))

	i := 1
	pow := func(x float32) float32 {
		y := math.Pow(float64(x), float64(i))
		if math.IsNaN(y) || math.IsInf(y, 0) {
			return 0
		}
		return float32(y)
	}
	points := make(plotter.XYs, 0, 8)
	// The stochastic gradient descent loop
	for i < 100 {
		start := time.Now()

		j := 0
		for j < 100 {
			pair := training[rnd.Intn(len(training))]
			input := make([]byte, len(pair.Question))
			copy(input, pair.Question)
			index := rand.Intn(len(pair.Answer))
			input = append(input, pair.Answer[:index]...)
			if len(input) < len(Indexes) {
				continue
			}
			entropy := MutualSelfEntropyUnitVector(db, input)
			for key, value := range entropy {
				inputs.X[j*256+key] = float32(value)
				outputs.X[j*256+key] = 0
			}
			outputs.X[j*256+int(pair.Answer[index])] = 1
			j++
		}
		// Calculate the gradients
		total := tf32.Gradient(cost).X[0]

		// Update the point weights with the partial derivatives using adam
		b1, b2 := pow(B1), pow(B2)
		for j, w := range set.Weights {
			for k, d := range w.D {
				g := d
				m := B1*w.States[StateM][k] + (1-B1)*g
				v := B2*w.States[StateV][k] + (1-B2)*g*g
				w.States[StateM][k] = m
				w.States[StateV][k] = v
				mhat := m / (1 - b1)
				vhat := v / (1 - b2)
				set.Weights[j].X[k] -= Eta * mhat / (float32(math.Sqrt(float64(vhat))) + 1e-8)
			}
		}

		// Housekeeping
		end := time.Since(start)
		fmt.Println(i, total, end)
		set.Zero()

		if math.IsNaN(float64(total)) {
			fmt.Println(total)
			break
		}

		set.Zero()
		others.Zero()

		points = append(points, plotter.XY{X: float64(i), Y: float64(total)})
		i++
	}

	// Plot the cost
	p := plot.New()

	p.Title.Text = "epochs vs cost"
	p.X.Label.Text = "epochs"
	p.Y.Label.Text = "cost"

	scatter, err := plotter.NewScatter(points)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "cost.png")
	if err != nil {
		panic(err)
	}

	//v := NewVectors("cc.en.300.vec.gz")
	//v.Test()

	//m := NewMatrix(0, 256, 256*256)
	//m.Data = m.Data[:256*256*256]
	//DirectSelfEntropyKernelParallel(m, m, m, Matrix{})
}
