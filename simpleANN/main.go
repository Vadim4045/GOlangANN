package main

import (
	"fmt"
	"os"
)

func main() {

	trainData, dim, err := GetMNISTdata(true)
	if err != nil {
		fmt.Println("Data load error:", err)
		os.Exit(1)
	}

	alfa := 0.5
	mu := 0.001
	config := []int{dim * dim, 200, 100, 10}

	nn, err := NewANN(alfa, mu, config)
	if err != nil {
		fmt.Println("Start error:", err)
		os.Exit(1)
	}

	err = nn.TrainNN(trainData, 1000, true)
	if err != nil {
		fmt.Println("Training error:", err)
	}

}
