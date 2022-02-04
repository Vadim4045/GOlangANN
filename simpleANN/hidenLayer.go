package main

import (
	"errors"
	"strconv"
)

type hidenLayer struct {
	alfa         float64
	layerContent []float64
	layerWeights [][]float64
}

func NewLayer(count, nextCount int, alfa float64) (newLayer hidenLayer) {

	newLayer.layerContent = make([]float64, count)
	newLayer.layerWeights = make([][]float64, nextCount)
	newLayer.alfa = alfa

	for i := range newLayer.layerWeights {
		newLayer.layerWeights[i] = make([]float64, count)
	}

	return
}

func (hl *hidenLayer) layerForvard(nextLayer *[]float64, alfa float64) error {
	if len(*nextLayer) != len(hl.layerWeights) {
		return errors.New("Slices length are not equals" +
			strconv.Itoa(len(*nextLayer)) + "/" + strconv.Itoa(len(hl.layerWeights)))
	}

	for idx1, c1 := range hl.layerWeights {
		var mySum float64 = 0
		for idx2, c2 := range hl.layerContent {
			mySum += c2 * c1[idx2]
		}
		(*nextLayer)[idx1] = sigmaFunc(mySum, hl.alfa)
	}

	return nil
}

func (hl *hidenLayer) layerBP(mis, layer []float64, mu float64) []float64 {

	res := make([]float64, len(hl.layerContent))

	for k := range hl.layerContent {
		for l := range hl.layerWeights {
			res[k] += mis[l] * hl.layerWeights[l][k]
		}
	}

	for k := range hl.layerContent {
		for l := range hl.layerWeights {
			hl.layerWeights[l][k] += mu * mis[l] * hl.layerContent[k] * layer[l]
		}
	}

	return res
}
