package main

import (
	"math"
	"time"
	"fmt"
	"os"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

func sum_ciphertext_vec(elements []*rlwe.Ciphertext, enc0 *rlwe.Ciphertext, eval *hefloat.Evaluator) (ct *rlwe.Ciphertext) {
	ct = enc0
	for _, i := range elements {
		ct, _ = eval.AddNew(i, ct)
	}

	return
}

func timing_aggregator_on_ciphertext_input(in int, params hefloat.Parameters, debug bool) (label int) {
	sk, pk, evk := keyGen(params)

	// receiving votes for an input in
	teachers := get_teachers()
	var votes []*rlwe.Ciphertext
	var weights []int
	var added []*rlwe.Ciphertext
	quota := 100
	enc0 := encrypt(0, pk, params)
	for j, tchr := range teachers {
		weights = append(weights, int(math.Floor(1/tchr.ft)))
		i := 0
		enc_vote := encrypt(float64(tchr.votes[in]), pk, params)
		for i <= weights[j] && i < quota {
			votes = append(votes, enc_vote)
			i++
		}
		if quota > weights[j]+1 {
			encAdded := encrypt(float64(quota-weights[j]-1), pk, params)
			added = append(added, encAdded)
			for i = 0; i < quota-weights[j]-1; i++ {
				votes = append(votes, enc0)
			}
		} else {
			added = append(added, enc0)
		}
	}

	// aggregator receiv votes and added
	start := time.Now()
	eval := hefloat.NewEvaluator(params, evk)
	v_sum := sum_ciphertext_vec(votes, enc0, eval)
	sum_added := sum_ciphertext_vec(added, enc0, eval)
	len_votes := complex(float64(len(votes)), 0.0)
	sub, _ := eval.SubNew(sum_added, len_votes)
	half := complex(0.5, 0)
	average_, _ := eval.MulRelinNew(sub, half)
	voted_label, _ := eval.AddNew(v_sum, average_)
	elapsed := time.Since(start)

	if debug {
		values := decrypt(sk, params, voted_label)

		if values[0] > 0 {
			label = 1
		} else {
			label = 0
		}
	} else {
		file.WriteString("=> Aggregation on ciphertext took : " + elapsed.String() + "\n")
	}

	return
}

func encode_weights(layer int, params hefloat.Parameters) (encWeights [][]*rlwe.Plaintext) {
	weights := get_weights(layer)

	for _, weight := range weights {
		var row []*rlwe.Plaintext
		for _, w := range weight {
			enc_w := encodeFloat(w, params)
			row = append(row, enc_w)
		}
		encWeights = append(encWeights, row)
	}

	return
}

func encode_bias(layer int, params hefloat.Parameters) (encBiases []*rlwe.Plaintext) {
	biases := get_bias(layer)

	for _, b := range biases {
		encBiases = append(encBiases, encodeFloat(b, params))
	}

	return
}

func encrypt_input(in int, pk *rlwe.PublicKey, params hefloat.Parameters) (enc_input []*rlwe.Ciphertext) {
	input := get_weights(0)[in]

	for _, in := range input {
		enc_input = append(enc_input, encrypt(in, pk, params))
	}

	return
}

func ctNeuron(input []*rlwe.Ciphertext, weights [][]*rlwe.Plaintext, bias []*rlwe.Plaintext, pk *rlwe.PublicKey, enc0 *rlwe.Ciphertext, eval *hefloat.Evaluator) (output []*rlwe.Ciphertext) {

	for col := 0; col < len(weights[0]); col++ {
		som := enc0
		for line := 0; line < len(weights); line++ {
			p, _ := eval.MulRelinNew(input[line], weights[line][col])
			som, _ = eval.AddNew(som, p)
		}
		som, _ = eval.AddNew(som, bias[col])
		
		eval.Rescale(som, som)
		output = append(output, som)
	}

	return
}

func ctRelu(input []*rlwe.Ciphertext, pk *rlwe.PublicKey, params hefloat.Parameters, eval *hefloat.Evaluator) (output []*rlwe.Ciphertext) {

	for _, x := range input {
		output = append(output, RelUEval(x, pk, params, eval))
	}

	return
}

func ctSigmoid(input []*rlwe.Ciphertext, pk *rlwe.PublicKey, params hefloat.Parameters, eval *hefloat.Evaluator) (output []*rlwe.Ciphertext) {

	for _, x := range input {
		output = append(output, SigmoidEval(x, pk, params, eval))
	}
	return
}

func infering_on_ciphertext_input(in int, params hefloat.Parameters, debug bool) (label int) {
	sk, pk, evk := keyGen(params)

	// defining the model
	layer1_weights := encode_weights(1, params)
	layer2_weights := encode_weights(2, params)
	layer3_weights := encode_weights(3, params)
	layer1_bias := encode_bias(1, params)
	layer2_bias := encode_bias(2, params)
	layer3_bias := encode_bias(3, params)

	// get input
	input := encrypt_input(in, pk, params)

	enc0 := encrypt(0, pk, params)
	eval := hefloat.NewEvaluator(params, evk)

	// stats inference
	start := time.Now()
	output_layer1 := ctNeuron(input, layer1_weights, layer1_bias, pk, enc0, eval)
	output_layer1 = ctRelu(output_layer1, pk, params, eval)

	output_layer2 := ctNeuron(output_layer1, layer2_weights, layer2_bias, pk, enc0, eval)
	output_layer2 = ctRelu(output_layer2, pk, params, eval)

	output_layer3 := ctNeuron(output_layer2, layer3_weights, layer3_bias, pk, enc0, eval)
	output_layer3 = ctSigmoid(output_layer3, pk, params, eval)

	elapsed := time.Since(start)

	if debug {
		enc_label := output_layer3[0]
		dec_label := decrypt(sk, params, enc_label)
		label = nearest_intger(dec_label[0])
	} else {
		file.WriteString("=> Inference on ciphertext took : " + elapsed.String() + "\n")
	}

	return
}

func verbose_infering_on_ciphertext(in int, params hefloat.Parameters, debug bool) (label int) {
	sk, pk, evk := keyGen(params)

	// defining the model
	layer1_weights := encode_weights(1, params)
	layer2_weights := encode_weights(2, params)
	layer3_weights := encode_weights(3, params)
	layer1_bias := encode_bias(1, params)
	layer2_bias := encode_bias(2, params)
	layer3_bias := encode_bias(3, params)

	// get input
	input := encrypt_input(in, pk, params)

	enc0 := encrypt(0.0, pk, params)
	eval := hefloat.NewEvaluator(params, evk)

	// stats inference
	start := time.Now()
	output_layer1 := ctNeuron(input, layer1_weights, layer1_bias, pk, enc0, eval)
	fmt.Println("******* Output layer 1 ****************** ")
	dec_out_layer1 := make([]float64, len(output_layer1))
	for i, o := range output_layer1 {
		val := decrypt(sk, params, o)[0]
		dec_out_layer1[i] = val
	}
	fmt.Println(dec_out_layer1)
	output_layer1 = ctRelu(output_layer1, pk, params, eval)
	fmt.Println("==> After ReLU")
	for i, o := range output_layer1 {
		val := decrypt(sk, params, o)[0]
		dec_out_layer1[i] = val
	}
	fmt.Println(dec_out_layer1)
	

	output_layer2 := ctNeuron(output_layer1, layer2_weights, layer2_bias, pk, enc0, eval)
	fmt.Println("******* Output layer 2 ****************** ")
	dec_out_layer2 := make([]float64, len(output_layer2))
	for i, o := range output_layer2 {
		val := decrypt(sk, params, o)[0]
		dec_out_layer2[i] = val
	}
	fmt.Println(dec_out_layer2)
	output_layer2 = ctRelu(output_layer2, pk, params, eval)
	fmt.Println("==> After ReLU")
	for i, o := range output_layer2 {
		val := decrypt(sk, params, o)[0]
		dec_out_layer2[i] = val
	}
	fmt.Println(dec_out_layer2)

	output_layer3 := ctNeuron(output_layer2, layer3_weights, layer3_bias, pk, enc0, eval)
	fmt.Println("******* Output layer 3 ****************** ")
	dec_out_layer3 := make([]float64, len(output_layer3))
	for i, o := range output_layer3 {
		val := decrypt(sk, params, o)[0]
		dec_out_layer3[i] = val
	}
	fmt.Println(dec_out_layer3)
	output_layer3 = ctSigmoid(output_layer3, pk, params, eval)
	fmt.Println("==> After Sigmoid")
	for i, o := range output_layer3 {
		val := decrypt(sk, params, o)[0]
		dec_out_layer3[i] = val
	}
	fmt.Println(dec_out_layer3)
	os.Exit(1)
	elapsed := time.Since(start)

	if debug {
		enc_label := output_layer3[0]
		dec_label := decrypt(sk, params, enc_label)
		label = nearest_intger(dec_label[0])
	} else {
		file.WriteString("=> Inference on ciphertext took : " + elapsed.String() + "\n")
	}

	return
}


