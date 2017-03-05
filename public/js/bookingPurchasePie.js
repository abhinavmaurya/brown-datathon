/**
 * Created by Akshaya on 3/5/2017.
 */

function plot2(){
    var width = 960,
        height = 500,
        radius = Math.min(width, height) / 2;

    var color = d3.scale.category10();
        // .range(["#98abc5", "#8a89a6", "#7b6888", "#6b486b", "#a05d56", "#d0743c", "#ff8c00"]);

    var arc = d3.svg.arc()
        .outerRadius(radius - 10)
        .innerRadius(0);

    var labelArc = d3.svg.arc()
        .outerRadius(radius - 100)
        .innerRadius(radius - 40);

    var pie = d3.layout.pie()
        .sort(null)
        .value(function(d) { return d.Count; });

    var svg = d3.select("body").append("svg")
        .attr("width", width)
        .attr("height", height)
        .append("g")
        .attr("transform", "translate(" + width / 2 + "," + height / 2 + ")");

    var color = d3.scale.ordinal()
        .range(["#deebf7", "#9ecae1", "#4292c6", "#084594"]);

    var tooltip = d3.select("body")
        .append("div")
        .attr("class", "tooltip")
        .style("opacity", 0);

    d3.csv("public/BookingPurchase.csv", type, function(error, data) {
        if (error) throw error;

        var g = svg.selectAll(".arc")
            .data(pie(data))
            .enter().append("g")
            .attr("class", "arc")
            .on("mouseover", function(d) {
            tooltip.transition()
                .duration(200)
                .style("opacity", .9);
            tooltip .html(d.data.Count + "<br/>")
                .style("left", (d3.event.pageX) + "px")
                .style("top", (d3.event.pageY + 10) + "px");
            })
            .on("mouseout", function(d) {
                tooltip.transition()
                    .duration(500)
                    .style("opacity", 0);

            });

        g.append("path")
            .attr("d", arc)
            .style("fill", function(d) { return color(d.data.Label); });

        g.append("text")
            .attr("transform", function(d) { return "translate(" + labelArc.centroid(d) + ")"; })
            .attr("dy", "-.40em")
            .attr("dx","-20")
            .style("font-size","15px")
            .style("font-type","Comic Sans")
            .text(function(d) { return d.data.Label; });
    });

    function type(d) {
        d.Count = +d.Count;
        return d;
    }
}

plot2();