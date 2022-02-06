package main

import (
	"fmt"
	"os"
)

var lern bool = true

func main() {
	var lernEpochs int = 1000

	if !lern {
		lernEpochs = 1
	}

	trainData, dim, err := GetMNISTdata(lern)
	if err != nil {
		fmt.Println("Data load error:", err)
		os.Exit(1)
	}

	alfa := 0.5
	mu := 0.001
	config := []int{dim * dim, 250, 150, 100, 10}
	//config := []int{dim * dim, 50, 100, 10}

	nn, err := NewANN(alfa, mu, config)
	if err != nil {
		fmt.Println("Start error:", err)
		os.Exit(1)
	}

	err = nn.TrainNN(trainData, lernEpochs, lern)
	if err != nil {
		fmt.Println("Training error:", err)
	}

}
