let places_visited = ['IN', 'AE', 'NP', 'GB', 'VN', 'BD'];
svg_countries = document.querySelectorAll("svg path");
// console.log(svg_countries);

for(let i = 0; i < places_visited.length; i++){
	for(let j = 0; j < svg_countries.length; j++){
		// svg_countries[j].style.stroke = 'rgb(255, 255, 255)';
		if(places_visited[i].toString() == (svg_countries[j].id).toString()){
			if (!svg_countries[j].hasAttribute('style')) {
                svg_countries[j].setAttribute('style', '');
            }

            svg_countries[j].style.fill = 'rgb(100, 100, 200)';
		}
	}
}

let tooltip = document.getElementById("tooltip");

svg_countries.forEach(function(country) {
	country.addEventListener('mouseover', function(event) {
		// Retrieve the title attribute
		let title = country.getAttribute('title');

		// Set tooltip content
		tooltip.innerHTML = title;

		// Calculate the position relative to the cursor
		const offsetX = 10;
		const offsetY = 10;
		const tooltipX = event.pageX + offsetX;
		const tooltipY = event.pageY + offsetY;

		// Set tooltip position
		tooltip.style.left = tooltipX + 'px';
		tooltip.style.top = tooltipY + 'px';

		// Display tooltip
		tooltip.style.display = 'block';
	});

	country.addEventListener('mouseout', function() {
		// Hide tooltip on mouseout
		tooltip.style.display = 'none';
	});
});