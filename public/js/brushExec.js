/**
 * Created by Akshaya on 3/5/2017.
 */

function plot1() {
    var margin = {top: 10, right: 10, bottom: 100, left: 40},
        margin2 = {top: 430, right: 10, bottom: 20, left: 40},
        width = 960 - margin.left - margin.right,
        height = 500 - margin.top - margin.bottom,
        height2 = 500 - margin2.top - margin2.bottom;

    var parseDate = d3.time.format("%d-%b-%Y").parse;

    var x = d3.time.scale().range([0, width]),
        x2 = d3.time.scale().range([0, width]),
        y = d3.scale.linear().range([height, 0]),
        y2 = d3.scale.linear().range([height2, 0]);

    var xAxis = d3.svg.axis().scale(x).orient("bottom"),
        xAxis2 = d3.svg.axis().scale(x2).orient("bottom"),
        yAxis = d3.svg.axis().scale(y).orient("left");

    var brush = d3.svg.brush()
        .x(x2)
        .on("brush", brushed);

    var area = d3.svg.area()
        .interpolate("monotone")
        .x(function (d) {
            return x(d.Day);
        })
        .y0(height)
        .y1(function (d) {
            return y(d.RecordCount);
        });

    var area2 = d3.svg.area()
        .interpolate("monotone")
        .x(function (d) {
            return x2(d.Day);
        })
        .y0(height2)
        .y1(function (d) {
            return y2(d.RecordCount);
        });

    var svg = d3.select("#plot1").append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom);

    svg.append("defs").append("clipPath")
        .attr("id", "clip")
        .append("rect")
        .attr("width", width)
        .attr("height", height);

    var focus = svg.append("g")
        .attr("class", "focus")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    var context = svg.append("g")
        .attr("class", "context")
        .attr("transform", "translate(" + margin2.left + "," + margin2.top + ")");

    var zoom = d3.behavior.zoom()
        .on("zoom", draw);

// Add rect cover the zoomed graph and attach zoom event.
    var rect = svg.append("svg:rect")
        .attr("class", "pane")
        .attr("width", width)
        .attr("height", height)
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
        .call(zoom);

//    d3.csv("sp500.csv", type, function(error, data) {
    d3.csv("public/sp_date.csv", type, function (error, data) {
//        console.log(data);
        x.domain(d3.extent(data.map(function (d) {
//            d3.time.format("%Y-%m-%d").parseDate;
//            console.log(parseDate(d.Day));
//             console.log(d.Day);
            return d.Day;
        })));
        y.domain([0, d3.max(data.map(function (d) {
            // console.log(d);
            // console.log("fucker" + d.RecordCount);
            return d.RecordCount;
        }))]);
        x2.domain(x.domain());
        y2.domain(y.domain());

        // Set up zoom behavior
        zoom.x(x);

        svg.append("text")
            .attr("class", "y label")
            .attr("text-anchor", "end")
            .attr("y", 6)
            .attr("dy", "1.90em")
            .attr("transform", "rotate(-90)")
            .text("Number of records");

        focus.append("path")
            .datum(data)
            .attr("class", "area")
            .attr("d", area);

        focus.append("g")
            .attr("class", "x axis")
            .attr("transform", "translate(0," + height + ")")
            .call(xAxis);

        focus.append("g")
            .attr("class", "y axis")
            .call(yAxis);

        context.append("path")
            .datum(data)
            .attr("class", "area")
            .attr("d", area2);

        context.append("g")
            .attr("class", "x axis")
            .attr("transform", "translate(0," + height2 + ")")
            .text("Date")
            .call(xAxis2);

        context.append("g")
            .attr("class", "x brush")
            .call(brush)
            .selectAll("rect")
            .attr("y", -6)
            .attr("height", height2 + 7);

        // brush.on('brush', function (d) {
        //     k = brush.extent();
        //     j = data.filter(function (d) {
        //         return k[0] <= d.Day && k[1] >= d.Day;
        //     });
        //     // newBarChart(j);
        // });

    });

    function brushed() {
        x.domain(brush.empty() ? x2.domain() : brush.extent());

        focus.select(".area").attr("d", area);
        focus.select(".x.axis").call(xAxis);
        // Reset zoom scale's domain
        zoom.x(x);
    }

    function draw() {
        focus.select(".area").attr("d", area);
        focus.select(".x.axis").call(xAxis);
        // Force changing brush range
        brush.extent(x.domain());
        var list = [];
        list.push(x.domain()[0]);
        list.push(x.domain()[1]);
        // console.log(list);
        plot(list);
        svg.select(".brush").call(brush);


    }

    function type(d) {
        // console.log("sup" + parseDate(d.Day));
        d.Day = parseDate(d.Day);
        d.RecordCount = +d.RecordCount;
        return d;
    }

}


function plot(list){
    // console.log(list);
    d3.csv("datathon_tadata.csv", function (error, data1) {
        cf= crossfilter(data1);
        byDate = cf.dimension(function(d){return d.day;});


    });

}

plot1();