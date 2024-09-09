package main

import (
	"log"
	"math"

	//"fmt"
	//"os"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	//"github.com/tuneinsight/lattigo/v5/he"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	//"github.com/tuneinsight/lattigo/v5/utils"
	"github.com/tuneinsight/lattigo/v5/utils/bignum"
)

func initScheme(logn, logq0, logscale, L, logp int) (params hefloat.Parameters) {
	logq := make([]int, L+1)
	logq[0] = logq0
	for i := 1; i < L+1; i++ {
		logq[i] = logscale
	}
	var err error

	if params, err = hefloat.NewParametersFromLiteral(
		hefloat.ParametersLiteral{
			LogN:            logn,        // A ring degree of 2^{logn}
			LogQ:            logq,        // An initial prime of 55 bits and L primes of 45 bits
			LogP:            []int{logp}, // The log2 size of the key-switching prime
			LogDefaultScale: logscale,    // The default log2 of the scaling factor
		}); err != nil {
		panic(err)
	}

	return
}

func keyGen(params hefloat.Parameters) (sk *rlwe.SecretKey, pk *rlwe.PublicKey, evk *rlwe.MemEvaluationKeySet) {
	kgen := rlwe.NewKeyGenerator(params)

	sk = kgen.GenSecretKeyNew()
	pk = kgen.GenPublicKeyNew(sk)

	rlk := kgen.GenRelinearizationKeyNew(sk) // relinearization key
	evk = rlwe.NewMemEvaluationKeySet(rlk)   // evaluation key

	return
}

func encrypt(mypt float64, pk *rlwe.PublicKey, params hefloat.Parameters) (ct *rlwe.Ciphertext) {
	pt := encodeFloat(mypt, params)

	enc := rlwe.NewEncryptor(params, pk)

	ct, err := enc.EncryptNew(pt)
	if err != nil {
		panic(err)
	}

	return
}

func decrypt(sk *rlwe.SecretKey, params hefloat.Parameters, ct *rlwe.Ciphertext) (values []float64) {
	dec := rlwe.NewDecryptor(params, sk)

	pt := dec.DecryptNew(ct)
	values = make([]float64, 1)

	enc := hefloat.NewEncoder(hefloat.Parameters(params))
	err := enc.Decode(pt, values)
	if err != nil {
		log.Fatal(err)
	}
	return
}

func encodeFloat(myfloat float64, params hefloat.Parameters) (pt *rlwe.Plaintext) {
	pt = hefloat.NewPlaintext(params, params.MaxLevel())

	pt_slice := make([]float64, 1)
	pt_slice[0] = myfloat

	ecd := hefloat.NewEncoder(hefloat.Parameters(params))

	if err := ecd.Encode(pt_slice, pt); err != nil {
		panic(err)
	}

	return
}

func RelUEval(ct *rlwe.Ciphertext, pk *rlwe.PublicKey, params hefloat.Parameters, eval *hefloat.Evaluator) *rlwe.Ciphertext {
	prec := params.EncodingPrecision()

	SiLU := func(x float64) (y float64) {
		return math.Max(0, x)
	}

	interval := bignum.Interval{
		Nodes: 31,
		A:     *bignum.NewFloat(-60, prec),
		B:     *bignum.NewFloat(60, prec),
	}

	poly := bignum.ChebyshevApproximation(SiLU, interval)

	polyEval := hefloat.NewPolynomialEvaluator(params, eval)

	scalarmul, scalaradd := poly.ChangeOfBasis()

	res, err := eval.MulNew(ct, scalarmul)
	if err != nil {
		panic(err)
	}

	if err = eval.Add(res, scalaradd, res); err != nil {
		panic(err)
	}

	if err = eval.Rescale(res, res); err != nil {
		panic(err)
	}

	if res, err = polyEval.Evaluate(res, poly, params.DefaultScale()); err != nil {
		panic(err)
	}

	return res
}

func ReLUEvalpt(f float64, params hefloat.Parameters) float64 {
	prec := params.EncodingPrecision()
	ReLU := func(x float64) (y float64) {
		return x / (math.Exp(-x) + 1)
	}

	interval := bignum.Interval{
		Nodes: 31,
		A:     *bignum.NewFloat(-60, prec),
		B:     *bignum.NewFloat(60, prec),
	}

	poly := bignum.ChebyshevApproximation(ReLU, interval)

	want := complex(f, 0)
	tmp := bignum.NewComplex().SetPrec(prec)
	want = poly.Evaluate(tmp.SetComplex128(want)).Complex128()
	return real(want)
}

func SigmoidEval(ct *rlwe.Ciphertext, pk *rlwe.PublicKey, params hefloat.Parameters, eval *hefloat.Evaluator) *rlwe.Ciphertext {
	prec := params.EncodingPrecision()

	Sigmoid := func(x float64) (y float64) {
		return 1 / (math.Exp(-x) + 1)
	}

	interval := bignum.Interval{
		Nodes: 15,
		A:     *bignum.NewFloat(-30, prec),
		B:     *bignum.NewFloat(30, prec),
	}

	poly := bignum.ChebyshevApproximation(Sigmoid, interval)

	scalarmul, scalaradd := poly.ChangeOfBasis()

	res, err := eval.MulNew(ct, scalarmul)
	if err != nil {
		panic(err)
	}

	if err = eval.Add(res, scalaradd, res); err != nil {
		panic(err)
	}

	if err = eval.Rescale(res, res); err != nil {
		panic(err)
	}

	polyEval := hefloat.NewPolynomialEvaluator(params, eval)

	if res, err = polyEval.Evaluate(res, poly, params.DefaultScale()); err != nil {
		panic(err)
	}

	return res
}

func SigmoidEvalpt(f float64, params hefloat.Parameters) float64 {
	prec := params.EncodingPrecision()
	Sigmoid := func(x float64) (y float64) {
		return 1 / (math.Exp(-x) + 1)
	}

	interval := bignum.Interval{
		Nodes: 15,
		A:     *bignum.NewFloat(-30, prec),
		B:     *bignum.NewFloat(30, prec),
	}

	poly := bignum.ChebyshevApproximation(Sigmoid, interval)

	want := complex(f, 0)
	tmp := bignum.NewComplex().SetPrec(prec)
	want = poly.Evaluate(tmp.SetComplex128(want)).Complex128()
	return real(want)
}

func EvalPow(ct *rlwe.Ciphertext, n int, eval *hefloat.Evaluator, params hefloat.Parameters) (res *rlwe.Ciphertext) {

	res, _ = eval.MulRelinNew(ct, ct)
	if err := eval.Rescale(res, res); err != nil {
		panic(err)
	}

	/* fmt.Printf("Level ct1 : %d | %f ( vs %f) \n", res.Level(), &res.Scale.Value, &ct.Scale.Value)
	fmt.Printf("Level ct1 : %d | %f ( vs %f) \n", ct.Level(), &ct.Scale.Value, &ct.Scale.Value) */
	for i := 0; i < n-2; i++ {
		res, _ = eval.MulRelinNew(ct, res)
		if err := eval.Rescale(res, res); err != nil {
			panic(err)
		}

	}

	//fmt.Printf("Level res : %d | %f\n", res.Level(), &res.Scale.Value)
	return
}

func ApproxSigmoid(ct *rlwe.Ciphertext, pk *rlwe.PublicKey, params hefloat.Parameters, eval *hefloat.Evaluator) *rlwe.Ciphertext {
	constante := encrypt(0.5, pk, params)
	coef1 := encrypt(0.197, pk, params)
	coef2 := encrypt(-0.004, pk, params)

	res1, _ := eval.MulRelinNew(ct, ct)
	res1, _ = eval.MulRelinNew(res1, ct)
	res1, _ = eval.MulRelinNew(coef2, res1) // -0.004*x^3

	res2, _ := eval.MulRelinNew(coef1, ct) // 0.197*x

	res, _ := eval.AddNew(res1, res2)
	res, _ = eval.AddNew(res, constante)

	return res
}

func ApproxRelu(ct *rlwe.Ciphertext, pk *rlwe.PublicKey, params hefloat.Parameters, eval *hefloat.Evaluator) *rlwe.Ciphertext {
	sig := ApproxSigmoid(ct, pk, params, eval)

	res, _ := eval.MulRelinNew(ct, sig) // x*sigmoid(x)
	return res
}
