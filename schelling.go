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
	"flag"
	"fmt"
	"github.com/grd/stat"
	"log"
	"math"
	"math/rand"
	"os"
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

func (r modelRun) String() string {
	return fmt.Sprintf("%d,%d,%d,%f,%d,%d,%d", r.runNumber, r.size, r.vision, r.tolerance, r.initGroups, r.finalGroups, r.ticks)
}

type modelRuns []modelRun

// declare global variables
var w *bufio.Writer
var verbose = false
var writeToFile = true
var vision int
var tolerance float64
var filename string

func aggregateRuns(numRuns, size, vision int, tolerance float64, verbose bool) {
	// Set up environment, perform the desired number of runs,
	// and output summary statistics
	// TODO: This function is doing a lot of work, individual model runs
	// could probably be shunted into a modelRun() function.

	// setup measurement variables
	successes := 0
	times := make(stat.IntSlice, numRuns)       //only used for stat
	initGroups := make(stat.IntSlice, numRuns)  //only used for stat
	finalGroups := make(stat.IntSlice, numRuns) //only used for stat
	results := make([]modelRun, numRuns)        //aggregate model outcomes

	// open file if necessary
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

	// execute runs
	for run := 0; run < numRuns; run++ {
		//model setup
		model := setup(size)
		r := modelRun{
			runNumber:   run + 1,
			size:        size,
			vision:      vision,
			tolerance:   tolerance,
			initGroups:  countDistinct(model),
			finalGroups: -1,
			ticks:       -1}

		ticks := int64(0)
		if verbose {
			fmt.Println(model)
			fmt.Printf("%d distinct groups at start\n", r.initGroups)
		}

		// model run
		for !isConverged(model) {
			step(model)
			ticks++
			if int64(ticks) > int64(500*len(model)) { // arbitary number to avoid infinite loops
				if verbose {
					fmt.Println("Model failed to stabilize")
				}
				break
			}
		}

		if isConverged(model) {
			r.finalGroups = countDistinct(model)
			if verbose {
				fmt.Printf("%d distinct groups at end after %d moves\n", r.finalGroups, ticks)
			}
			r.ticks = ticks
			successes++
		}

		results[run] = r //add run outcomes to total results

		// write to file
		if writeToFile {
			_, err := w.WriteString(fmt.Sprintln(r))
			if err != nil {
				log.Fatal(err)
			}
		}
	}

	// output statistics to console
	fmt.Println("Summary statistics:")

	// populating IntSlices for statistics
	for i := 0; i < len(results); i++ {
		times[i] = results[i].ticks
		initGroups[i] = results[i].initGroups
		finalGroups[i] = results[i].finalGroups
	}

	fmt.Printf("%d runs reach equilibrium (%.1f%%) in %.1f ticks (s.d.: %.1f)\n", successes,
		100*float64(successes)/float64(numRuns), stat.Mean(times), stat.Sd(times))
	fmt.Printf("%.1f average initial groups (s.d.: %.1f)\n", stat.Mean(initGroups), stat.Sd(initGroups))
	fmt.Printf("%.1f average final groups (s.d.: %.1f)\n", stat.Mean(finalGroups), stat.Sd(finalGroups))

	return
}

func countDistinct(model []float64) int64 {
	// Identify coherent subpopulations, what Brandt et al call "firewalls."

	val := model[0]
	x := int64(0)

	for _, element := range model {
		if val != element {
			val = element
			x++
		}
	}

	if model[0] == model[len(model)-1] { // wrap around
		x--
	}

	return x
}

func setup(size int) []float64 {
	// Return an initialized 1-D Schelling model, a slice of ints limited
	// to the range [0, 1] of an arbitary size.

	model := make([]float64, size)
	for i := range model {
		model[i] = float64(rand.Intn(2))
	}
	return model
}

func isConverged(model []float64) bool {
	// Return true if all agents in the model are happy, else return false.

	for idx := range model {
		if !isHappy(model, idx) {
			return false
		}
	}

	return true
}

func isHappy(model []float64, idx int) bool {
	// Return true if the proportion of nearby agents of the same type is greater than or equal to
	// its tolerance threshold. The number of agents examined is given by the vision global variable.

	count := 0
	for x := 1; x <= vision; x++ {
		y := int(math.Mod(float64(idx-x), float64(len(model))))
		if y < 0 {
			y += len(model)
		}
		count += int(model[y])

		y = int(math.Mod(float64(idx+x), float64(len(model))))
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

func step(model []float64) {
	// Using random activation, find an unhappy agent and
	// tell it to move.

	idx := rand.Intn(len(model)) // might need to be len - 1?

	// cycle until you find an unhappy agent
	for isHappy(model, idx) {
		idx = rand.Intn(len(model))
	}
	move(model, idx)
}

func move(model []float64, idx int) {
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
	// initialize model variables from console input
	var numAgents, numRuns int

	flag.IntVar(&numAgents, "s", 0, "number of agents in the model")
	flag.IntVar(&numRuns, "n", 0, "number of model runs")
	flag.IntVar(&vision, "w", 0, "neighborhood size")
	flag.Float64Var(&tolerance, "t", 0, "agent tolerance")
	flag.BoolVar(&verbose, "v", false, "verbose console output")
	flag.StringVar(&filename, "o", "", "filename to write to, if necessary")
	flag.Parse()

	// input validation
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
	if filename == "" {
		writeToFile = false
	}

	aggregateRuns(numRuns, numAgents, vision, tolerance, verbose)
}
