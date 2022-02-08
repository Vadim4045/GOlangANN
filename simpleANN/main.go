package main

import (
	"fmt"
	"os"
	"os/signal"
	"sync"
	"syscall"
)

var wg sync.WaitGroup
var lern bool = true
var myGo bool = true
var alfa float64 = 0.5
var mu float64 = 0.04

func main() {
	fmt.Printf("pid: %d\n", os.Getpid())

	var lernEpochs int = 1000

	if !lern {
		lernEpochs = 1
	}

	trainData, dim, err := GetMNISTdata(lern)
	if err != nil {
		fmt.Println("Data load error:", err)
		os.Exit(1)
	}

	config := []int{dim * dim, 300, 200, 10}

	nn, err := NewANN(alfa, mu, config)
	if err != nil {
		fmt.Println("Start error:", err)
		os.Exit(1)
	}

	signalChanel := make(chan os.Signal, 1)

	signal.Notify(signalChanel,
		syscall.SIGINT,
		syscall.SIGUSR1,
		syscall.SIGUSR2)

	go func() {
		for {
			s := <-signalChanel
			switch s {

			case syscall.SIGINT:
				fmt.Println(" Prepare to stop.")
				myGo = false

			case syscall.SIGUSR1:
				mu *= 2
				fmt.Printf("\n MU multiplayed.( new MU is %f )\n", mu)

			case syscall.SIGUSR2:
				mu /= 2
				fmt.Printf("\n MU divided.( new MU is %f )\n", mu)
			}
		}
	}()

	err = nn.TrainNN(trainData, lernEpochs, lern)
	if err != nil {
		fmt.Println("Training error:", err)
	}

}
