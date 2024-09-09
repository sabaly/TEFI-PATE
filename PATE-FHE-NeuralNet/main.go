package main

import (
	"fmt"
	"log"
	"os"

	"math"

	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	//"github.com/tuneinsight/lattigo/v5/utils/bignum"
)

var file *os.File

func test(params hefloat.Parameters) {
	//var err error
	sk, pk, evk := keyGen(params)

	//prec := params.EncodingPrecision()

	//LogSlots := params.LogMaxSlots()
	//Slots := 1 << LogSlots

	for f := 0.0; f < 8; f += 1.0 {
		//want := complex(f, 0)

		ct1 := encrypt(float64(f), pk, params)

		eval := hefloat.NewEvaluator(params, evk) // evaluator
		ct1 = EvalPow(ct1, 2, eval, params)
		//fmt.Println("==> level : ", ct1.Level())
		res := RelUEval(ct1, pk, params, eval)
		dec_res := decrypt(sk, params, res)[0]

		wante_res := ReLUEvalpt(math.Pow(f, 2), params)

		fmt.Printf("ReLU(enc(%f)) = %f", math.Pow(f, 2), dec_res)
		fmt.Printf(" vs ReLU(%f) = %f\n", math.Pow(f, 2), wante_res)
	}
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
func main() {
	filePath := "results"
	var err error
	file, err = os.OpenFile(filePath, os.O_CREATE|os.O_WRONLY, 0666)
	if err != nil {
		log.Println(err)
	}
	L := 20
	params := initScheme(12, 35, 25, L, 30) // 12, 35, 25, L, 41 and 14, 55, 45, L, 61
	//test(params)

	fmt.Print("Timing tests are running................")
	file.WriteString("//---------------------------------------\n")
	file.WriteString("// TIMING TESTS\n")
	file.WriteString("//---------------------------------------\n")
	timing_aggregator_on_plaintext_input(0, false)
	timing_aggregator_on_ciphertext_input(0, params, false)
	infering_on_plaintext_input(1, false)
	infering_on_ciphertext_input(0, params, false)
	fmt.Println("Done")

	fmt.Print("Accuracy tests are running..............\n")
	file.WriteString("//---------------------------------------\n")
	file.WriteString("// ACCURACY TESTS\n")
	file.WriteString("//---------------------------------------\n")
	//inputs := get_weights(0)
	test_size := 100 //int(0.001 * float64(len(inputs)))
	test_size_str := fmt.Sprintf("%d", test_size)
	file.WriteString("=> Test size : " + test_size_str + "\n")
	pt_infered_labels := make([]int, test_size)
	ct_infered_labels := make([]int, test_size)
	for i := 0; i < test_size; i++ {
		pt_infered_labels = append(pt_infered_labels, infering_on_plaintext_input(i, true))
	}
	fmt.Println("Infering  on ciphertext starts")
	for i := 0; i < test_size; i++ {
		ct_infered_labels = append(ct_infered_labels, infering_on_ciphertext_input(i, params, true))
		fmt.Print("\r==> ", i+1, "%")
	}
	fmt.Println()
	acc := accuracy(pt_infered_labels, ct_infered_labels)
	acc_str := fmt.Sprintf("%.2f", acc*100)
	file.WriteString(acc_str + " % Inference with plaintext and inference with ciphertext are equal.\n")
	fmt.Println(".......................Done")

	//-----------------------------------------------------------------------
	// Autres tests
	//-----------------------------------------------------------------------
	/* for i := 0; i < 20; i++ {
		fmt.Print(infering_on_plaintext_input(i, true), " ")
	}
	fmt.Println()
	for i := 0; i < 20; i++ {
		fmt.Print(infering_on_ciphertext_input(i, params, true), " ")
	}
	fmt.Println() */
}
