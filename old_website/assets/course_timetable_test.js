var NUM_SLOTS = 5;

timetable_layers = {
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
			"",
			""
		], 
		"Friday" : [
			"",
			"",
			"",
			"",
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
		"Monday" : [], 
		"Tuesday" : [], 
		"Wednesday" : [], 
		"Thursday" : [], 
		"Friday" : [], 
		"Saturday" : []
	},
}

id_mappings = {
	"Monday" : ["MN1", "MN2", "MN3", "MN4", "MN5"],
	"Tuesday" : ["TU1", "TU2", "TU3", "TU4", "TU5"],
	"Wednesday" : ["WD1", "WD2", "WD3", "WD4", "WD5"],
	"Thurday" : ["TH1", "TH2", "TH3", "TH4", "TH5"],
	"Friday" : ["FR1", "FR2", "FR3", "FR4", "FR5"],
	"Saturday" : ["ST1", "ST2", "ST3", "ST4", "ST5"]
};

function updateTimetable(){
	for(var i in id_mappings){
		for(var j = 0; j < NUM_SLOTS; j++){
			console.log(id_mappings[i][j]);
			document.getElementById(id_mappings[i][j]).innerHTML = timetable_layers["base"][j];
		}
	}
}