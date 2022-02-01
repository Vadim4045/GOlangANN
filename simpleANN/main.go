package main

import (
	"fmt"
	"os"
	"time"
)

func main() {
	start := time.Now()

	trainData, dim, err := GetMNISTdata(true, 0)
	if err != nil {
		fmt.Println("Data load error: ", err)
		os.Exit(1)
	}

	config := []int{dim * dim, 20, 20, 10}
	nn, err := NewANN(config)
	if err != nil {
		fmt.Println("Start error: ", err)
		os.Exit(1)
	}

	err = nn.TrainNN(trainData, 100)
	if err != nil {
		fmt.Println("Training error: ", err)
	}

	fmt.Println(time.Now().Sub(start))
}
