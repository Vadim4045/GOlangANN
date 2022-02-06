package main

import (
	"errors"
	"math"
	"strings"
)

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

func sigmaFunc(x, alfa float64) (res float64) {
	return 1.0 / (1.0 + math.Exp(-x*alfa))
	/* if x >= 0 {
		return x
	} else {
		return 0.01 * x
	} */
}

func sigmoidPrime(x, alfa float64) float64 {
	return alfa * x * (1.0 - x)
	/* if x > 0 {
		return 1.0
	} else {
		return -0.01
	} */
}

func strPrefRemove(str, pref string) (string, error) {
	for count := len(str); count > 0; count-- {
		res := strings.TrimPrefix(str, pref)
		if res == str {
			return res, nil
		} else {
			str = res
		}
	}
	return "", errors.New("Trim error")
}
