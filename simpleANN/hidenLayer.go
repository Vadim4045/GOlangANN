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
		wg.Add(1)
		go func(i int, arr []float64) {

			var mySum float64 = 0
			for idx2, c2 := range hl.layerContent {
				mySum += c2 * arr[idx2]
			}

			(*nextLayer)[i] = sigmaFunc(mySum, hl.alfa)

			wg.Done()

		}(idx1, c1)

	}

	wg.Wait()

	return nil
}

func (hl *hidenLayer) layerBP(mis, layer []float64, mu float64) []float64 {

	res := make([]float64, len(hl.layerContent))

	for k := range hl.layerContent {
		wg.Add(1)
		go func(i int) {
			for l := range hl.layerWeights {
				res[i] += mis[l] * hl.layerWeights[l][i]
			}

			wg.Done()
		}(k)

	}

	wg.Wait()

	for k := range hl.layerContent {
		wg.Add(1)
		go func(i int) {
			for l := range hl.layerWeights {
				hl.layerWeights[l][i] += mu * mis[l] * hl.layerContent[i] * layer[l]
			}

			wg.Done()
		}(k)
	}

	wg.Wait()

	return res
}
