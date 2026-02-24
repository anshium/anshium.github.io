const places_visited = ['IN', 'AE', 'NP', 'GB', 'VN', 'BD'];
const svg_countries = document.querySelectorAll("#world_map path");

places_visited.forEach(code => {
    const country = document.getElementById(code);
    if (country) {
        country.style.fill = '#4a9eff'; // Minimalist highlight blue
    }
});

const tooltip = document.getElementById("tooltip");

svg_countries.forEach(function (country) {
    country.addEventListener('mouseover', function (event) {
        const title = country.getAttribute('title');
        if (!title) return;

        tooltip.innerHTML = title;
        tooltip.style.display = 'block';

        const offsetX = 15;
        const offsetY = 15;
        tooltip.style.left = (event.pageX + offsetX) + 'px';
        tooltip.style.top = (event.pageY + offsetY) + 'px';
    });

    country.addEventListener('mousemove', function (event) {
        const offsetX = 15;
        const offsetY = 15;
        tooltip.style.left = (event.pageX + offsetX) + 'px';
        tooltip.style.top = (event.pageY + offsetY) + 'px';
    });

    country.addEventListener('mouseout', function () {
        tooltip.style.display = 'none';
    });
});
