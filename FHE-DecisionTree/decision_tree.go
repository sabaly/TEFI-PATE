package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strconv"
	"time"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"gorgonia.org/vecf32"
)

type node struct {
	children_left  int
	children_right int
	feature        float64
	threshold      float64
	value          []int
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

func get_children(left bool) (children []int) {
	filename := ""
	if left {
		filename = "./docs/children_left.csv"
	} else {
		filename = "./docs/children_right.csv"
	}

	records := read_csv(filename)

	for _, label := range records {
		p, _ := strconv.ParseFloat(label[0], 64)
		children = append(children, int(p))
	}

	return
}

func get_feature_threshold(feature bool) (result []float64) {
	filename := ""
	if feature {
		filename = "./docs/feature.csv"
	} else {
		filename = "./docs/threshold.csv"
	}

	records := read_csv(filename)

	for _, label := range records {
		p, _ := strconv.ParseFloat(label[0], 64)
		result = append(result, p)
	}

	return
}

func get_value() (value [][]float32) {
	filename := "./docs/values.csv"

	records := read_csv(filename)

	for _, r := range records {
		var row []float32
		for _, in := range r {
			w, _ := strconv.ParseFloat(in, 32)
			row = append(row, float32(w))
		}
		value = append(value, row)
	}

	return
}

func get_data() (value [][]float64) {
	filename := "./docs/data.csv"

	records := read_csv(filename)

	for _, r := range records {
		var row []float64
		for _, in := range r {
			w, _ := strconv.ParseFloat(in, 64)
			row = append(row, w)
		}
		value = append(value, row)
	}

	return
}

type intDoubleSlice [][]int

func (l *intDoubleSlice) pop() (int, int) {
	length := len(*l)
	lastEle := (*l)[length-1]
	*l = (*l)[:length-1]
	return lastEle[0], lastEle[1]
}

func get_tree() {
	children_left := get_children(true)
	children_right := get_children(false)
	feature := get_feature_threshold(true)
	threshold := get_feature_threshold(false)
	value := get_value()
	fmt.Println(children_left)
	fmt.Println(children_right)
	fmt.Println(feature)
	fmt.Println(threshold)
	fmt.Println(value)

	node_depth := make([]int, len(feature))
	is_leaves := make([]bool, len(feature))
	var stack intDoubleSlice
	stack = append(stack, []int{0, 0})

	for len(stack) > 0 {
		node_id, depth := stack.pop()
		node_depth[node_id] = depth

		is_split_node := (children_left[node_id] != children_right[node_id])
		if is_split_node {
			stack = append(stack, []int{children_left[node_id], depth + 1})
			stack = append(stack, []int{children_right[node_id], depth + 1})
		} else {
			is_leaves[node_id] = true
		}
	}

}

func inference_plaintext(input []float64) int {
	// get the tree
	children_left := get_children(true)
	children_right := get_children(false)
	feature := get_feature_threshold(true)
	threshold := get_feature_threshold(false)
	value := get_value()

	node_depth := make([]int, len(feature))
	var stack intDoubleSlice
	var label int
	stack = append(stack, []int{0, 0})

	for len(stack) > 0 {
		node_id, depth := stack.pop()
		node_depth[node_id] = depth

		is_split_node := (children_left[node_id] != children_right[node_id])
		if is_split_node {
			if input[int(feature[node_id])] <= threshold[node_id] {
				stack = append(stack, []int{children_left[node_id], depth + 1})
			} else {
				stack = append(stack, []int{children_right[node_id], depth + 1})
			}
		} else {
			label = vecf32.Argmax(value[node_id])
			//fmt.Println(value[node_id])
		}
	}

	return label
}

func mult_vec(a []*rlwe.Ciphertext, b []*rlwe.Ciphertext, eval *hefloat.Evaluator) (ct *rlwe.Ciphertext) {
	tmp := make([]*rlwe.Ciphertext, len(a))
	for i := 0; i < len(tmp); i++ {
		tmp[i], _ = eval.MulRelinNew(a[i], b[i])
		//eval.Rescale(tmp[i], tmp[i])
	}
	ct = tmp[0]
	for i := 1; i < len(tmp); i++ {
		eval.Add(ct, tmp[i], ct)
	}

	eval.Rescale(ct, ct)

	return
}

func mult_scal_vec(a *rlwe.Ciphertext, b []*rlwe.Ciphertext, eval *hefloat.Evaluator) (res []*rlwe.Ciphertext) {
	tmp := make([]*rlwe.Ciphertext, len(b))

	for i := 0; i < len(tmp); i++ {
		tmp[i], _ = eval.MulRelinNew(a, b[i])
		if err := eval.Rescale(tmp[i], tmp[i]); err != nil {
			panic(err)
		}
	}

	res = tmp
	return
}

func Enc_Predict(input []*rlwe.Ciphertext,
	node_id int, children_left []int,
	children_right []int,
	feature [][]*rlwe.Ciphertext,
	threshold []*rlwe.Ciphertext,
	value [][]*rlwe.Ciphertext,
	pk *rlwe.PublicKey,
	params hefloat.Parameters,
	eval *hefloat.Evaluator,
	deep float64,
	verbose bool) (res []*rlwe.Ciphertext) {
	is_split_node := (children_left[node_id] != children_right[node_id])
	if is_split_node {
		cx_feat := mult_vec(input, feature[node_id], eval)
		ct1, _ := eval.SubNew(cx_feat, threshold[node_id])
		ct2, _ := eval.SubNew(threshold[node_id], cx_feat)
		s1 := EvalSign(ct1, pk, params, eval)
		s2 := EvalSign(ct2, pk, params, eval)

		if verbose {
			d_ct1 := decrypt(g_sk, params, ct1)
			d_ct2 := decrypt(g_sk, params, ct2)
			d_s1 := decrypt(g_sk, params, s1)[0]
			d_s2 := decrypt(g_sk, params, s2)[0]
			d_ct1 = append(d_ct1, d_s1)
			d_ct2 = append(d_ct2, d_s2)
			fmt.Println("s_ct1 --> ", d_ct1)
			fmt.Println("s_ct2 --> ", d_ct2, "\n_________________")
		}

		tmp1 := mult_scal_vec(s1,
			Enc_Predict(input, children_left[node_id], children_left, children_right, feature, threshold, value, pk, params, eval, deep-1, verbose), eval)
		tmp2 := mult_scal_vec(s2,
			Enc_Predict(input, children_right[node_id], children_left, children_right, feature, threshold, value, pk, params, eval, deep-1, verbose), eval)
		tmp := make([]*rlwe.Ciphertext, len(tmp1))
		for i := 0; i < len(tmp); i++ {
			tmp[i], _ = eval.AddNew(tmp1[i], tmp2[i])
		}
		if verbose {
			dec_tmp1 := make([]float32, len(tmp1))
			for i := 0; i < len(tmp1); i++ {
				dec_tmp1[i] = float32(decrypt(g_sk, params, tmp1[i])[0])
			}
			fmt.Print("Dec_tmp1 : ", dec_tmp1, " | ")
			dec_tmp2 := make([]float32, len(tmp2))
			for i := 0; i < len(tmp2); i++ {
				dec_tmp2[i] = float32(decrypt(g_sk, params, tmp2[i])[0])
			}
			fmt.Print("Dec_tmp2 : ", dec_tmp2, " | ")
		}
		for i := 0; i < len(tmp); i++ {
			tmp[i], _ = eval.AddNew(tmp[i], value[node_id][i])
		}
		res = tmp
		if verbose {
			dec_res := make([]float32, len(res))
			for i := 0; i < len(res); i++ {
				dec_res[i] = float32(decrypt(g_sk, params, res[i])[0])
			}
			fmt.Println("Dec_res : ", dec_res)
		}
	} else {
		res = value[node_id]
	}

	return
}

func inference_ciphertext(input []*rlwe.Ciphertext,
	pk *rlwe.PublicKey,
	params hefloat.Parameters,
	eval *hefloat.Evaluator,
	verbose bool) []*rlwe.Ciphertext {
	children_left := get_children(true)
	children_right := get_children(false)
	init_feature := get_feature_threshold(true)
	threshold := get_feature_threshold(false)
	value := get_value()

	// encoding feature
	feature := make([][]int, len(init_feature))
	for ind, f := range init_feature {
		vec := make([]int, len(input))
		for i := 0; i < len(input); i++ {
			if i == int(f) {
				vec[i] = 1
			} else {
				vec[i] = 0
			}
		}
		feature[ind] = vec
	}

	// encrypt : threshold
	enc_threshold := make([]*rlwe.Ciphertext, len(threshold))
	for i, th := range threshold {
		enc_threshold[i] = encrypt(float64(th), pk, params)
	}
	// encrypt : value
	enc_value := make([][]*rlwe.Ciphertext, len(value))
	for i, val := range value {
		tmp := make([]*rlwe.Ciphertext, len(val))
		for j, v := range val {
			tmp[j] = encrypt(float64(v), pk, params)
		}
		enc_value[i] = tmp
	}
	// encrypt : feature
	enc_feature := make([][]*rlwe.Ciphertext, len(feature))
	for i, feat := range feature {
		tmp := make([]*rlwe.Ciphertext, len(feat))
		for j, v := range feat {
			tmp[j] = encrypt(float64(v), pk, params)
		}
		enc_feature[i] = tmp
	}
	start := time.Now()
	max_depth := 4.0
	res := Enc_Predict(input, 0, children_left, children_right, enc_feature, enc_threshold, enc_value, pk, params, eval, max_depth, verbose)
	elapsed := time.Since(start)
	if verbose {
		fmt.Println("Inference on ciphertext took : ", elapsed.String())
	}

	return res
}

func accuracy(pt_infered []int, ct_infered []int) (acc float64) {
	counts := 0.0
	for i, label := range ct_infered {
		if label == pt_infered[i] {
			counts++
		}
	}

	acc = counts / float64(len(ct_infered))
	return
}

func accuracy_test() {
	data := get_data()[:1]
	//fmt.Println(data)

	fmt.Print("Inference on plaintext......")
	pt_inferences := make([]int, len(data))

	start := time.Now()
	for i := 0; i < len(data); i++ {
		input := data[i]
		pt_inferences[i] = inference_plaintext(input)
	}
	elapsed := time.Since(start)
	fmt.Println("Done...", elapsed.String())

	// init params
	L := 11
	params := initScheme(12, 22, 20, L, 30) // 14, 55, 45, L, 61 and 12, 35, 25, L, 41

	sk, pk, evk := keyGen(params)
	eval := hefloat.NewEvaluator(params, evk)

	fmt.Print("Inference on ciphertext.....")
	ct_inferences := make([]int, len(data))
	start = time.Now()
	for i := 0; i < len(data); i++ {
		input := data[i]
		enc_input := make([]*rlwe.Ciphertext, len(input))
		for i, in := range input {
			enc_input[i] = encrypt(float64(in), pk, params)
		}

		res := inference_ciphertext(enc_input, pk, params, eval, false)

		dec_res := make([]float32, len(res))
		for i := 0; i < len(res); i++ {
			dec_res[i] = float32(decrypt(sk, params, res[i])[0])
		}

		ct_inferences[i] = vecf32.Argmax(dec_res)
	}
	elapsed = time.Since(start)
	fmt.Println("Done...", elapsed.String())

	//fmt.Println(ct_inferences)

	acc := accuracy(pt_inferences, ct_inferences)

	fmt.Println("Accuracy : ", acc*100, " %")
}
