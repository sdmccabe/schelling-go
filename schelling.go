package main

// Schelling 1D Model
// Ported from Python to Go
// Stefan McCabe

// This is an implementation of the one-dimensional Schelling segregation model, developed
// as practice writing ABMs in Go and to test possible optimizations.  It builds on an
// implementation of the 1-D Schelling model I wrote in Python in early 2015. When writing
// that model, I generally adhered to the formalized version of the model described in the
// following citation:
//
// Brandt, C., Immorlica, N., Kamath, G., & Kleinberg, R. (2012).
// An analysis of one-dimensional Schelling segregation.
// In STOC '12 Proceedings of the forty-fourth annual ACM symposium
// on theory of computing (p. 789). ACM Press.
// doi:10.1145/2213977.2214048

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"github.com/grd/stat"
	"github.com/pkg/profile"
	"log"
	"math/rand"
	"os"
	"runtime"
	"sync"
	"time"
)

// declare data types
type modelRun struct {
	runNumber   int
	size        int
	vision      int
	tolerance   float64
	initGroups  int64
	finalGroups int64
	ticks       int64
}

type modelRuns []modelRun
type model []int

func (r modelRun) String() string {
	return fmt.Sprintf("%d,%d,%d,%f,%d,%d,%d", r.runNumber, r.size, r.vision, r.tolerance, r.initGroups, r.finalGroups, r.ticks)
}

func (m model) String() string {
	var buffer bytes.Buffer

	for _, x := range m {
		if x == 0 {
			buffer.WriteString("X")
		} else if x == 1 {
			buffer.WriteString("O")
		} else {
			fmt.Println("Error: Unexpected model element")
			os.Exit(1)
		}
	}

	return buffer.String()
}

// declare global variables
var profileRun bool
var w *bufio.Writer
var verbose bool
var writeToFile bool
var vision int
var tolerance float64
var filename string
var parallel bool
var numChunks int

func aggregateRuns(numRuns, size, vision int, tolerance float64, verbose bool) {
	// Set up environment, perform the desired number of runs,
	// and output summary statistics

	// set up measurement variables
	successes := 0
	times := make(stat.IntSlice, 0)       //only used for stat
	initGroups := make(stat.IntSlice, 0)  //only used for stat
	finalGroups := make(stat.IntSlice, 0) //only used for stat

	// numChunks := runtime.NumCPU() * 2
	chunkSize := numRuns / numChunks
	results := make(chan modelRun, numChunks+1)

	if writeToFile {
		f, err := os.Create(filename)
		defer f.Close()
		w = bufio.NewWriter(f)
		defer w.Flush()

		//TODO: Writing csv headers is very fragile, see if this can be improved.
		_, err = w.WriteString("run,size,vision,tolerance,init.blocks,final.blocks,ticks\n")
		if err != nil {
			log.Fatal(err)
		}
	}
	if parallel {
		go func() {
			for {
				result := <-results
				if result.ticks != -1 {
					successes++
				}
				times = append(times, result.ticks)
				initGroups = append(initGroups, result.initGroups)
				finalGroups = append(finalGroups, result.finalGroups)
				if writeToFile {
					w.WriteString(fmt.Sprintln(result))
				}
			}
		}()
	}
	var wg sync.WaitGroup
	if parallel {
		wg.Add(numChunks)
		for i := 0; i < numChunks; i++ {
			go func(n, s int) {
				for j := 0; j < n; j++ {
					results <- runModel(s)
				}
				wg.Done()
			}(chunkSize, size)
		}

		wg.Wait() // wait for all model runs to end before computing statistics
	} else {
		serialResults := make([]modelRun, numRuns)
		for i := 0; i < numRuns; i++ {
			serialResults[i] = runModel(size)
		}
		// populating IntSlices for statistics
		for i := 0; i < len(serialResults); i++ {
			times = append(times, serialResults[i].ticks)
			if times[i] != -1 {
				successes++
			}
			initGroups = append(initGroups, serialResults[i].initGroups)
			finalGroups = append(finalGroups, serialResults[i].finalGroups)
		}

	}

	// output statistics to console
	fmt.Println("Summary statistics:")
	fmt.Printf("%d runs reach equilibrium (%.1f%%) in %.1f ticks (s.d.: %.1f)\n", successes,
		100*float64(successes)/float64(numRuns), stat.Mean(times), stat.Sd(times))
	fmt.Printf("%.1f average initial groups (s.d.: %.1f)\n", stat.Mean(initGroups), stat.Sd(initGroups))
	fmt.Printf("%.1f average final groups (s.d.: %.1f)\n", stat.Mean(finalGroups), stat.Sd(finalGroups))
}

func runModel(size int) modelRun {
	// Execute one run of the model. Return true if the model converged.

	// model setup
	model := setup(size)
	r := modelRun{
		size:        size,
		vision:      vision,
		tolerance:   tolerance,
		initGroups:  countDistinct(model),
		finalGroups: -1,
		ticks:       -1}

	ticks := int64(1)
	if verbose {
		fmt.Printf("Run number %d\n", r.runNumber)
		fmt.Printf("%d distinct groups at start\n", r.initGroups)
		fmt.Println(model)
	}

	// model run
	for !isConverged(model) {
		step(model)
		ticks++
		if verbose {
			fmt.Println(model)
		}
		if int64(ticks) > int64(500*len(model)) { // arbitary number to avoid infinite loops
			if verbose {
				fmt.Println("Model failed to stabilize")
			}
			ticks = -1
			break
		}
	}

	success := isConverged(model)
	if success {
		r.finalGroups = countDistinct(model)
		if verbose {
			//fmt.Println(model)
			fmt.Printf("%d distinct groups at end after %d moves\n", r.finalGroups, ticks)
			fmt.Println()
		}
		r.ticks = ticks
	}

	return r
}

func countDistinct(model model) int64 {
	// Identify coherent subpopulations, what Brandt et al call "firewalls."

	val := model[0]
	x := int64(0)

	for _, element := range model {
		if val != element {
			val = element
			x++
		}
	}

	if model[0] != model[len(model)-1] { // wrap around
		x++
	}

	return x
}

func setup(size int) model {
	// Return an initialized 1-D Schelling model, a slice of ints limited
	// to the range [0, 1] of an arbitary size.

	m := make(model, size)
	for i := range m {
		m[i] = rand.Intn(2)
	}
	return m
}

func isConverged(model model) bool {
	// Return true if all agents in the model are happy, else return false.

	for idx := range model {
		if !isHappy(model, idx) {
			return false
		}
	}

	return true
}

func isHappy(model model, idx int) bool {
	// Return true if the proportion of nearby agents of the same type is greater than or equal to
	// its tolerance threshold. The number of agents examined is given by the vision global variable.

	count := 0
	for x := 1; x <= vision; x++ {
		y := (idx - x) % len(model)
		if y < 0 {
			y += len(model)
		}
		count += int(model[y])

		y = (idx + x) % len(model)
		if y < 0 {
			y += len(model)
		}
		count += int(model[y])
	}

	if model[idx] == 0 { // invert for agents of type zero
		count = 2*vision - count
	}

	neighbors := float64(count) / float64((2 * vision))
	if neighbors < tolerance {
		return false
	}
	return true
}

func step(model model) {
	// Using random activation, find an unhappy agent and
	// tell it to move.

	idx := rand.Intn(len(model))

	// cycle until you find an unhappy agent
	for isHappy(model, idx) {
		idx = rand.Intn(len(model))
	}
	move(model, idx)
}

func move(model model, idx int) {
	// Move an unhappy agent to new places in the model at random until it is happy.
	// TODO: Some method of tracking unhappy users could reduce randomness here.
	// TODO: IIRC, this is slightly more random than the Brandt model. Update comment with clarification.

	tries := 0
	unhappy := true

	// arbitary number of tries to avoid infinite loops
	for unhappy && tries < (2*len(model)) {

		val := model[idx]                             // store the agent type
		model = append(model[:idx], model[idx+1:]...) // delete the model index
		idx = rand.Intn(len(model))                   // randomly generate a new index

		// the next three lines insert the agent into the new index
		model = append(model, 0)
		copy(model[idx+1:], model[idx:])
		model[idx] = val

		tries++
		unhappy = !isHappy(model, idx) // evaluate the agent's happiness at the new location
	}
}

func main() {
	// seed RNG
	rand.Seed(time.Now().UTC().UnixNano())

	// initialize model variables from console input
	var numAgents, numRuns int

	flag.IntVar(&numAgents, "s", 0, "number of agents in the model")
	flag.IntVar(&numRuns, "n", 0, "number of model runs")
	flag.IntVar(&vision, "w", 0, "neighborhood size")
	flag.Float64Var(&tolerance, "t", 0, "agent tolerance")
	flag.BoolVar(&verbose, "v", false, "verbose console output")
	flag.StringVar(&filename, "o", "", "filename to write to, if necessary")
	flag.IntVar(&numChunks, "p", runtime.NumCPU(), "number of chunks to split the runs into. set to 0 for serial")
	flag.BoolVar(&profileRun, "profile", false, "profile application run")
	flag.Parse()

	// input validation
	if profileRun {
		defer profile.Start(profile.CPUProfile, profile.ProfilePath(".")).Stop()
	}
	if numChunks == 0 {
		fmt.Println("number of chunks must be greater than zero. enter one for serial.")
	} else if numChunks == 1 {
		parallel = false
	} else {
		parallel = true
		fmt.Printf("GOMAXPROCS = %d\n", runtime.NumCPU())
	}
	if numAgents <= 0 {
		fmt.Println("Please enter the number of agents to simulate.")
		os.Exit(1)
	}
	if numRuns <= 0 {
		fmt.Println("Please enter the number of model runs to be performed.")
		os.Exit(1)
	}
	if vision <= 0 {
		fmt.Println("Please enter the desired neighborhood size.")
		os.Exit(1)
	}
	if tolerance <= 0 || tolerance >= 1 {
		fmt.Println("Error: tolerance must be a decimal greater than zero and less than one.")
		os.Exit(1)
	}
	if vision > numAgents {
		fmt.Println("Error: vision cannot be greater than the number of agents.")
		os.Exit(1)
	}
	if verbose && parallel {
		fmt.Println("Error: verbose and parallel cannot be enabled at the same time.")
		os.Exit(1)
	}
	if filename == "" {
		writeToFile = false
	} else {
		writeToFile = true
	}

	aggregateRuns(numRuns, numAgents, vision, tolerance, verbose)
}
