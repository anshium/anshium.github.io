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
countdown('Auto1', 'Aug 24, 2023 23:55:00')
countdown('Auto2', 'Sep 4, 2023 23:55:00')
countdown('Algo1', 'Infinite')
countdown('Sci1', 'Aug 14, 2023 23:59:59')
countdown('OSN1', 'Aug 12, 2023 23:59:59')
countdown('OSN2', 'Aug 25, 2023 23:59:59')
countdown('ESW1', 'Aug 15, 2023 18:00:00')