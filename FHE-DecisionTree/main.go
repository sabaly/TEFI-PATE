package main

import (
	"fmt"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"gorgonia.org/vecf32"
)

var g_sk *rlwe.SecretKey

func test() {
	//fmt.Println(("Hello world !"))
	//get_tree()

	//input := []float64{5.9, 3.0, 5.1, 1.8}
	input := []float64{77, 0, 172744, 15, 10, 2, 0, 0, 4, 1, 0, 0, 35, 39}
	label := inference_plaintext((input))

	fmt.Println("infered label : ", label)

	fmt.Println("################")

	L := 11
	params := initScheme(14, 55, 45, L, 61) // 14, 55, 45, 61

	sk, pk, evk := keyGen(params)
	g_sk = sk
	eval := hefloat.NewEvaluator(params, evk)

	enc_input := make([]*rlwe.Ciphertext, len(input))
	for i, in := range input {
		enc_input[i] = encrypt(float64(in), pk, params)
	}

	res := inference_ciphertext(enc_input, pk, params, eval, true)

	dec_res := make([]float32, len(res))
	for i := 0; i < len(res); i++ {
		dec_res[i] = float32(decrypt(sk, params, res[i])[0])
	}
	fmt.Println("Decrypted vector : ", dec_res)

	fmt.Println("Decrypted label : ", vecf32.Argmax(dec_res))
}

func test_sign() {
	L := 7
	params := initScheme(14, 55, 45, L, 61)

	sk, pk, evk := keyGen(params)
	g_sk = sk
	eval := hefloat.NewEvaluator(params, evk)

	for i := -10.0; i <= 10.0; i += 1.0 {
		ct := encrypt(i, pk, params)
		res := EvalSign_bis(ct, pk, params, eval, 5)
		expected := 0
		if i <= 0 {
			expected = 1
		}
		d_res := decrypt(sk, params, res)[0]
		fmt.Println("Value : ", i, " | Expected : ", expected, " | EvalSign : ", d_res)
	}
}
func main() {
	//test()
	fmt.Println("############ Testing accuracy #########################")
	accuracy_test()

	//test_sign()
}
