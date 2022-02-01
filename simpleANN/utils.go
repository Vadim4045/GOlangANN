package main

import "math"

func maxInArray(arr []float64) (int, float64) {
	var max float64
	var res int
	for i, val := range arr {
		if i == 0 || val > max {
			max = val
			res = i
		}
	}
	return res, max
}

func minInArray(arr []float64) (int, float64) {
	var min float64
	var res int
	for i, val := range arr {
		if i == 0 || val < min {
			min = val
			res = i
		}
	}
	return res, min
}

func sigmaFunc(x float64) (res float64) {
	return 1.0 / (1.0 + math.Exp(-x*alfa))
	/* if x > 0 {
		return x
	} else {
		return 0.01 * x
	} */
}

func sigmoidPrime(x float64) float64 {
	return alfa * x * (1.0 - x)
	/* if x > 0 {
		return 1
	} else {
		return -0.01
	} */
}
