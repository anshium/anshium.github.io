var NUM_SLOTS = 5;

let timetable_layers = {
	base : {
		"Monday" : [
			"", 
			"DASS (H-105)", 
			"", 
			"Into to Human Sciences (H-105)", 
			""
		], 
		"Tuesday" : [
			"", 
			"", 
			"", 
			"Science 2 (H-205)", 
			""
		], 
		"Wednesday" : [
			"", 
			"Machine, Data and Learning (H-205)", 
			"VE (2)", 
			"", 
			""
		], 
		"Thursday" : [
			"", 
			"DASS (H-105)", 
			"",
			"Into to Human Sciences (H-105)",
			""
		], 
		"Friday" : [
			"",
			"",
			"",
			"Science 2 (H-205)",
			""
		], 
		"Saturday" : [
			"",
			"Machine, Data and Learning (H-205)",
			"",
			"",
			""
		]
	},
	computer_graphics : {
		"Monday" : ["", "", "", "", ""], 
		"Tuesday" : ["", "", "", "", ""],
		"Wednesday" : ["Computer Graphics (H-103)", "", "", "", ""],
		"Thursday" : ["", "", "", "", ""],
		"Friday" : ["", "", "", "", ""],
		"Saturday" : ["Computer Graphics (H-103)", "", "", "", ""],
	},
	topping : {
		"Monday" : ["", "", "", "", ""], 
		"Tuesday" : ["", "", "", "", ""],
		"Wednesday" : ["", "", "", "", ""],
		"Thursday" : ["", "", "", "", ""],
		"Friday" : ["", "", "", "", ""],
		"Saturday" : ["", "", "", "", ""],
	},
}

let timetable_colors = {
	base : ["black", "yellowgreen"],
	computer_graphics : ["white", "#1083e8"],
	topping : ["white", "white"]
};

let id_mappings = {
	"Monday" : ["MN1", "MN2", "MN3", "MN4", "MN5"],
	"Tuesday" : ["TU1", "TU2", "TU3", "TU4", "TU5"],
	"Wednesday" : ["WD1", "WD2", "WD3", "WD4", "WD5"],
	"Thursday" : ["TH1", "TH2", "TH3", "TH4", "TH5"],
	"Friday" : ["FR1", "FR2", "FR3", "FR4", "FR5"],
	"Saturday" : ["ST1", "ST2", "ST3", "ST4", "ST5"]
};

function updateTimetable(){
	for(let layer in timetable_layers){
		for(var i in id_mappings){
			for(var j = 0; j < NUM_SLOTS; j++){
				// console.log(timetable_layers[layer][i][j]);
				if(document.getElementById(id_mappings[i][j]).innerHTML == 0){
					document.getElementById(id_mappings[i][j]).innerHTML = timetable_layers[layer][i][j];
					document.getElementById(id_mappings[i][j]).style.color = timetable_colors[layer][0]; // bad way, need to have better design, but in a hurry (prob :))
					document.getElementById(id_mappings[i][j]).style.backgroundColor = timetable_colors[layer][1];
				} /*else {
					// PLS IGNORE FOR NOW
					console.log("Overlapping layer items error!");
				}*/
			}
		}
	}
}

updateTimetable();

let selected_layers = timetable_layers["base"];

// Get all checkbox elements with the class 'option'
const checkboxes = document.querySelectorAll('.option');

// Add event listener to each checkbox
checkboxes.forEach(checkbox => {
	checkbox.addEventListener('change', updateSelectedOptions);
});

function updateSelectedOptions(){

}

function countdown(id, time){
	// let time = document.getElementById(id); // Format: "Jan 5, 2024 15:37:25"
	if(time == "Infinite"){
		console.log("Here1");
		document.getElementById(id).innerHTML = "Infinite";
		return;
	}
	console.log("Here2");
	var countDownDate = new Date(time).getTime();

	// Update the count down every 1 second
	var x = setInterval(function() {
		// Get today's date and time
		var now = new Date().getTime();
		
		// Find the distance between now and the count down date
		var distance = countDownDate - now;

		// Time calculations for days, hours, minutes and seconds
		var days = Math.floor(distance / (1000 * 60 * 60 * 24));
		var hours = Math.floor((distance % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
		var minutes = Math.floor((distance % (1000 * 60 * 60)) / (1000 * 60));
		var seconds = Math.floor((distance % (1000 * 60)) / 1000);

		// Display the result in the element with id=\"id\"
		document.getElementById(id).innerHTML = days + "d " + hours + "h "
		+ minutes + "m " + seconds + "s ";
		// If the count down is finished, write some text
		if (distance < 0) {
			clearInterval(x);
			document.getElementById(id).innerHTML = "Subah ho gayi mamu?";
		}
	}, 1000);
}
countdown('Auto1', 'Aug 24, 2024 23:55:00')
countdown('Auto2', 'Sep 4, 2024 23:55:00')
countdown('Algo1', 'Infinite')
countdown('Sci1', 'Aug 16, 2024 23:59:59')
countdown('Sci2', 'Aug 23, 2024 23:59:59')
countdown('OSN1', 'Aug 12, 2023 23:59:59')
countdown('OSN2', 'Aug 25, 2024 23:59:59')
countdown('ESW1', 'Aug 17, 2024 23:59:59')

/*Some problem in deployment, this is a test.*/