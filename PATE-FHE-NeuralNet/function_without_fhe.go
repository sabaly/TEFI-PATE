package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"
	"time"
)

type Teacher struct {
	local_m float64
	ft      float64
	votes   []int
}

func read_csv(filename string) [][]string {
	file, err := os.Open(filename)

	if err != nil {
		log.Fatal("Error while reading the file", err)
	}

	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()

	if err != nil {
		fmt.Println("Error reading records")
	}

	return records
}

func get_teachers() []Teacher {
	// get teacher's parameters
	records := read_csv("docs/teachers.csv")

	teachers := make([]Teacher, len(records))
	for i, record := range records {
		teachers[i].local_m, _ = strconv.ParseFloat(record[1], 64)
		teachers[i].ft, _ = strconv.ParseFloat(record[2], 64)

		// get teachers votes
		file := "docs/teacher_" + strconv.Itoa(i) + "_preds.csv"
		//fmt.Println(file)
		preds := read_csv(file)
		for _, pred := range preds {
			p, _ := strconv.ParseFloat(pred[0], 64)
			teachers[i].votes = append(teachers[i].votes, int(p))
		}
	}

	return teachers
}

func voted_labels(train bool) (voted_labels []int) {
	file := "docs/voted_labels_"
	if train {
		file = file + "train.csv"
	} else {
		file = file + "test.csv"
	}
	records := read_csv(file)

	for _, label := range records {
		p, _ := strconv.ParseFloat(label[0], 64)
		voted_labels = append(voted_labels, int(p))
	}

	return
}

func sum(elements []int) int {
	som := 0
	for _, i := range elements {
		som += i
	}

	return som
}

func timing_aggregator_on_plaintext_input(in int, debug bool) (label int) {
	// receiving votes for an input in
	teachers := get_teachers()
	var votes []int
	var weights []int
	var added []int
	quota := 100
	for j, tchr := range teachers {
		weights = append(weights, int(math.Floor(1/tchr.ft)))
		i := 0
		for i <= weights[j] && i < quota {
			votes = append(votes, tchr.votes[in])
			i++
		}
		if quota > weights[j]+1 {
			added = append(added, quota-weights[j]-1)
			for i = 0; i < quota-weights[j]-1; i++ {
				votes = append(votes, 0)
			}
		} else {
			added = append(added, 0)
		}
	}

	// aggregator receiv votes and added
	start := time.Now()
	v_sum := sum(votes)
	average_ := (len(votes) - sum(added)) / 2
	if v_sum >= average_ {
		label = 1
	} else {
		label = 0
	}
	elapsed := time.Since(start)

	if debug == false {
		file.WriteString("=> Aggregation on plaintext took: " + elapsed.String() + "\n")
	}
	return
}

func get_weights(layer int) (weights [][]float64) {
	file := ""
	if layer != 0 {
		file = "docs/model_layer" + strconv.Itoa(layer) + "_weights.csv"
	} else {
		file = "docs/student_training_data.csv"
	}
	records := read_csv(file)

	for _, r := range records {
		var row []float64
		for _, in := range r {
			w, _ := strconv.ParseFloat(in, 64)
			row = append(row, w)
		}
		weights = append(weights, row)
	}

	return
}

func get_bias(layer int) (biases []float64) {
	file := "docs/model_layer" + strconv.Itoa(layer) + "_bias.csv"
	records := read_csv(file)

	for _, r := range records {
		b, _ := strconv.ParseFloat(r[0], 64)
		biases = append(biases, b)
	}

	return
}

func ptNeuron(input []float64, weights [][]float64, bias []float64) (output []float64) {
	for col := 0; col < len(weights[0]); col++ {
		som := float64(0)
		for line := 0; line < len(weights); line++ {
			som = som + input[line]*weights[line][col]
		}
		som = som + bias[col]
		output = append(output, som)
	}

	return
}

func ptRelu(input []float64) (output []float64) {
	for _, x := range input {
		output = append(output, math.Max(0.0, x))
	}

	return
}

func ptSigmoid(input []float64) (output []float64) {

	for _, x := range input {
		output = append(output, 1/(1+math.Exp(-x)))
	}
	return
}

func nearest_intger(x float64) int {
	if x-math.Floor(x) > 0.5 {
		return int(math.Floor(x) + 1)
	}

	return int(math.Floor(x))
}

func infering_on_plaintext_input(in int, debug bool) (label int) {
	// defining the model
	layer1_weights := get_weights(1)
	layer2_weights := get_weights(2)
	layer3_weights := get_weights(3)
	layer1_bias := get_bias(1)
	layer2_bias := get_bias(2)
	layer3_bias := get_bias(3)

	// get input
	records := get_weights(0) // get_weights function is better to extract training datas
	input := records[in]

	// stats inference
	start := time.Now()
	output_layer1 := ptNeuron(input, layer1_weights, layer1_bias)
	output_layer1 = ptRelu(output_layer1)

	output_layer2 := ptNeuron(output_layer1, layer2_weights, layer2_bias)
	output_layer2 = ptRelu(output_layer2)

	output_layer3 := ptNeuron(output_layer2, layer3_weights, layer3_bias)
	output_layer3 = ptSigmoid(output_layer3)

	elapsed := time.Since(start)

	if debug == false {
		file.WriteString("=> Inference on plaintext took : " + elapsed.String() + "\n")
	}
	label = nearest_intger(output_layer3[0])
	return
}

func verbose_inference_on_plaintext(in int, debug bool) (label int) {
	// defining the model
	layer1_weights := get_weights(1)
	layer2_weights := get_weights(2)
	layer3_weights := get_weights(3)
	layer1_bias := get_bias(1)
	layer2_bias := get_bias(2)
	layer3_bias := get_bias(3)

	// get input
	records := get_weights(0) // get_weights function is better to extract training datas
	input := records[in]

	/* fmt.Println(input)
	label = 1
	return  */

	// stats inference
	start := time.Now()
	output_layer1 := ptNeuron(input, layer1_weights, layer1_bias)
	fmt.Println("******* Output layer 1 ****************** ")
	fmt.Println(output_layer1)
	output_layer1 = ptRelu(output_layer1)

	output_layer2 := ptNeuron(output_layer1, layer2_weights, layer2_bias)
	fmt.Println("******* Output layer 2 ****************** ")
	fmt.Println(output_layer2)
	output_layer2 = ptRelu(output_layer2)

	output_layer3 := ptNeuron(output_layer2, layer3_weights, layer3_bias)
	fmt.Println("******* Output layer 3 ****************** ")
	fmt.Println(output_layer3)
	output_layer3 = ptSigmoid(output_layer3)

	elapsed := time.Since(start)

	if debug == false {
		file.WriteString("=> Inference on plaintext took : " + elapsed.String() + "\n")
	}
	label = nearest_intger(output_layer3[0])

	return
}
