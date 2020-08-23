// URL: https://observablehq.com/@d3/sunburst
// Title: Sunburst
// Author: D3 (@d3)
// Version: 214
// Runtime version: 1

const m0 = {
  id: "a601aba88046a626@214",
  variables: [
    {
      inputs: ["md"],
      value: (function(md){return(
md`# Sunburst

This radial space-filling visualization created by [John Stasko](https://www.cc.gatech.edu/gvu/ii/sunburst/) shows the cumulative values of subtrees. It is commonly used to visualize software packages (the size of source code within nested packages) and file systems (the size of files within nested folders). See also the [zoomable version](/@d3/zoomable-sunburst) and the [icicle diagram](/@d3/icicle).`
)})
    },
    {
      name: "chart",
      inputs: ["partition","data","d3","DOM","width","color","arc","format","autosize"],
      value: (function(partition,data,d3,DOM,width,color,arc,format,autosize)
{
  const root = partition(data);

  const svg = d3.select(DOM.svg(width, width))
      .style("width", "100%")
      .style("height", "auto")
      .style("padding", "10px")
      .style("font", "10px sans-serif")
      .style("box-sizing", "border-box");
  
  svg.append("g")
      .attr("fill-opacity", 0.6)
    .selectAll("path")
    .data(root.descendants().filter(d => d.depth))
    .enter().append("path")
      .attr("fill", d => { while (d.depth > 1) d = d.parent; return color(d.data.name); })
      .attr("d", arc)
    .append("title")
      .text(d => `${d.ancestors().map(d => d.data.name).reverse().join("/")}\n${format(d.value)}`);

  svg.append("g")
      .attr("pointer-events", "none")
      .attr("text-anchor", "middle")
    .selectAll("text")
    .data(root.descendants().filter(d => d.depth && (d.y0 + d.y1) / 2 * (d.x1 - d.x0) > 10))
    .enter().append("text")
      .attr("transform", function(d) {
        const x = (d.x0 + d.x1) / 2 * 180 / Math.PI;
        const y = (d.y0 + d.y1) / 2;
        return `rotate(${x - 90}) translate(${y},0) rotate(${x < 180 ? 0 : 180})`;
      })
      .attr("dy", "0.35em")
      .text(d => d.data.name);

  return autosize(svg.node());
}
)
    },
    {
      name: "data",
      inputs: ["d3"],
      value: (function(d3){return(
d3.json("https://raw.githubusercontent.com/d3/d3-hierarchy/v1.1.8/test/data/flare.json")
)})
    },
    {
      name: "partition",
      inputs: ["d3","radius"],
      value: (function(d3,radius){return(
data => d3.partition()
    .size([2 * Math.PI, radius])
  (d3.hierarchy(data)
    .sum(d => d.value)
    .sort((a, b) => b.value - a.value))
)})
    },
    {
      name: "color",
      inputs: ["d3","data"],
      value: (function(d3,data){return(
d3.scaleOrdinal(d3.quantize(d3.interpolateRainbow, data.children.length + 1))
)})
    },
    {
      name: "format",
      inputs: ["d3"],
      value: (function(d3){return(
d3.format(",d")
)})
    },
    {
      name: "width",
      value: (function(){return(
932
)})
    },
    {
      name: "radius",
      inputs: ["width"],
      value: (function(width){return(
width / 2
)})
    },
    {
      name: "arc",
      inputs: ["d3","radius"],
      value: (function(d3,radius){return(
d3.arc()
    .startAngle(d => d.x0)
    .endAngle(d => d.x1)
    .padAngle(d => Math.min((d.x1 - d.x0) / 2, 0.005))
    .padRadius(radius / 2)
    .innerRadius(d => d.y0)
    .outerRadius(d => d.y1 - 1)
)})
    },
    {
      name: "autosize",
      value: (function(){return(
function autosize(svg) {
  document.body.appendChild(svg);
  const box = svg.getBBox();
  document.body.removeChild(svg);
  svg.setAttribute("viewBox", `${box.x} ${box.y} ${box.width} ${box.height}`);
  return svg;
}
)})
    },
    {
      name: "d3",
      inputs: ["require"],
      value: (function(require){return(
require("d3@5")
)})
    }
  ]
};

const notebook = {
  id: "a601aba88046a626@214",
  modules: [m0]
};

export default notebook;
