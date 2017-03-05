/**
 * Created by saritajoshi on 3/5/17.
 */
function getData(){

console.log(list.length);
    d3.csv("public/datathon_tadata.csv", function (error, data) {
        cf= crossfilter(data);
        byDate = cf.dimension(function(d){return d.day;});
    });
}


getData();