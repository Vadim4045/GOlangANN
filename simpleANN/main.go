package main

import (
	"fmt"
	"os"
	"sync"
	"time"
)

var prevEpoch int = 0

const epochsPerStep = 100

func main() {
	start := time.Now()

	var wg sync.WaitGroup

	trainData, dim, err := GetMNISTdata(true, 0)
	if err != nil {
		fmt.Println("Data load error: ", err)
		os.Exit(1)
	}

	alfa := 0.5
	mu := 0.025

	config := []int{dim * dim, 20, 10}

	for k := 0; k < 6; k++ {

		for i := 1; i < 5; i++ {
			wg.Add(1)

			go func(j int) {
				nn, err := NewANN(alfa, mu*float64(j), config)
				if err != nil {
					fmt.Println("Start error: ", err)
					wg.Done()
					return
				}

				err = nn.TrainNN(trainData, epochsPerStep)
				if err != nil {
					fmt.Println("Training error: ", err)
				}
				wg.Done()
			}(i)
		}
		wg.Wait()
		fmt.Println(time.Now().Sub(start))
		prevEpoch += epochsPerStep
	}

}
