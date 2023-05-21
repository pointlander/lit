// Copyright 2023 The Lit Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// Squad is the squad training data
type Squad struct {
	Data []struct {
		Paragraphs []struct {
			Context string `json:"context"`
			Qas     []struct {
				Answers []struct {
					AnswerStart int    `json:"answer_start"`
					Text        string `json:"text"`
				} `json:"answers"`
				ID               string `json:"id"`
				IsImpossible     bool   `json:"is_impossible"`
				PlausibleAnswers []struct {
					AnswerStart int    `json:"answer_start"`
					Text        string `json:"text"`
				} `json:"plausible_answers,omitempty"`
				Question string `json:"question"`
			} `json:"qas"`
		} `json:"paragraphs"`
		Title string `json:"title"`
	} `json:"data"`
	Version string `json:"version"`
}
