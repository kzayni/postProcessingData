const datasets = [
  {
    kind: "clean",
    name: "Clean shape",
    path: "C00_Clean-Shape/C00_TC241_Airfoil/Case_241_Clean_shape.dat",
    xVariable: "CoordinateX_Iced",
    yVariable: "CoordinateY_Iced",
    lineWidth: 2,
    dash: "solid",
    color: "black"
  },
  {
    kind: "experimental",
    name: "Experimental",
    path: "E00_Experimental-Data/E00_TC241_MCCS/Case-241-MCCS_MODIFIED.dat",
    xVariable: "CoordinateX_Iced",
    yVariable: "CoordinateY_Iced",
    lineWidth: 3,
    dash: "solid",
    color: "black",
    mode: "lines+markers",
    markerSymbol: "diamond",
    markerColor: "pink",
    markerSize: 10,
    markerLineColor: "black",
    markerLineWidth: 1,
    markerEvery: 25
  },
  {
    kind: "participant",
    path: "001_POLIMO_CHAMPS/001_TC241_01/SOLUTION_VARIABLES_CASE_IPW1_241.dat",
    xVariable: "CoordinateX_Iced",
    yVariable: "CoordinateY_Iced",
    lineWidth: 3,
    dash: "solid",
    color: "blue"
  }
];

function getParticipantIdFromPath(path) {
  const folderName = path.split("/")[0];
  const match = folderName.match(/^(\d{3})/);

  if (!match) {
    return "Unknown";
  }

  return match[1];
}

function getDatasetName(dataset) {
  if (dataset.kind === "participant") {
    return `Participant ${getParticipantIdFromPath(dataset.path)}`;
  }

  return dataset.name;
}

function extractVariables(text) {
  const variablesMatch = text.match(/VARIABLES\s*=\s*([\s\S]*?)\bZONE\b/i);

  if (!variablesMatch) {
    throw new Error("Could not find VARIABLES section.");
  }

  const variablesText = variablesMatch[1];
  const matches = [...variablesText.matchAll(/"([^"]+)"/g)];

  return matches.map(match => match[1].trim());
}

function extractNumbers(text) {
  const matches = text.match(/[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][+-]?\d+)?/g);
  return matches ? matches.map(Number) : [];
}

function getDataSection(text) {
  const lines = text.split(/\r?\n/);
  const firstNumericLineIndex = lines.findIndex(line => /^\s*[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][+-]?\d+)?/.test(line));

  if (firstNumericLineIndex < 0) {
    throw new Error("No numeric data found.");
  }

  return lines.slice(firstNumericLineIndex).join("\n");
}

function getOrderedPointCount(text) {
  const match = text.match(/\bI\s*=\s*(\d+)/i);
  return match ? parseInt(match[1], 10) : null;
}

function isBlockPacking(text) {
  return /DATAPACKING\s*=\s*BLOCK/i.test(text);
}

function parseTecplotDat(text, xVariable, yVariable) {
  const variables = extractVariables(text);

  const xIndex = variables.indexOf(xVariable);
  const yIndex = variables.indexOf(yVariable);

  if (xIndex < 0) {
    throw new Error(`Variable not found: ${xVariable}`);
  }

  if (yIndex < 0) {
    throw new Error(`Variable not found: ${yVariable}`);
  }

  const dataText = getDataSection(text);
  const values = extractNumbers(dataText);
  const nVariables = variables.length;

  if (isBlockPacking(text)) {
    const nPoints = getOrderedPointCount(text);

    if (!nPoints) {
      throw new Error("BLOCK Tecplot file found, but I=... could not be read.");
    }

    const xStart = xIndex * nPoints;
    const yStart = yIndex * nPoints;

    const x = values.slice(xStart, xStart + nPoints);
    const y = values.slice(yStart, yStart + nPoints);

    return { x, y };
  }

  const nRows = Math.floor(values.length / nVariables);
  const x = [];
  const y = [];

  for (let row = 0; row < nRows; row++) {
    x.push(values[row * nVariables + xIndex]);
    y.push(values[row * nVariables + yIndex]);
  }

  return { x, y };
}

async function loadDataset(dataset) {
  const response = await fetch(dataset.path);

  if (!response.ok) {
    throw new Error(`Could not load ${dataset.path}. HTTP status: ${response.status}`);
  }

  const text = await response.text();
  const data = parseTecplotDat(text, dataset.xVariable, dataset.yVariable);

  const baseTrace = {
    x: data.x,
    y: data.y,
    mode: "lines",
    type: "scatter",
    name: getDatasetName(dataset),
    line: {
      width: dataset.lineWidth,
      dash: dataset.dash,
      color: dataset.color
    }
  };

  if (!dataset.markerEvery) {
    return [baseTrace];
  }

  const markerX = data.x.filter((_, index) => index % dataset.markerEvery === 0);
  const markerY = data.y.filter((_, index) => index % dataset.markerEvery === 0);

  const markerTrace = {
    x: markerX,
    y: markerY,
    mode: "markers",
    type: "scatter",
    name: `${getDatasetName(dataset)} markers`,
    showlegend: false,
    marker: {
      symbol: dataset.markerSymbol || "circle",
      color: dataset.markerColor || dataset.color,
      size: dataset.markerSize || 6,
      line: {
        color: dataset.markerLineColor || dataset.color,
        width: dataset.markerLineWidth || 0
      }
    }
  };

  return [baseTrace, markerTrace];
}

function computeRanges(traces) {
  const allX = traces.flatMap(trace => trace.x).filter(Number.isFinite);
  const allY = traces.flatMap(trace => trace.y).filter(Number.isFinite);

  const xMin = -0.0173187471892;//Math.min(...allX);
  const xMax = 0.0349345794985; //Math.max(...allX);
  const yMin = -0.0183502382957; //Math.min(...allY);
  const yMax = 0.0280971632045; //Math.max(...allY);

  const xPadFactor = 1.0;
  const yPadFactor = 1.0;

  const xPad = 0.0; //xPadFactor * (xMax - xMin);
  const yPad = 0.0; //yPadFactor * (yMax - yMin);

  return {
    x: [xMin - xPad, xMax + xPad],
    y: [yMin - yPad, yMax + yPad]
  };
}

function buildLayout(ranges) {
  return {
    title: {
      text: "IPW1 Case 241 - Ice Shape Comparison",
      x: 0.5,
      xanchor: "center",
      font: {
        size: 24
      }
    },

    font: {
      family: "Arial, Helvetica, sans-serif",
      size: 18
    },

    xaxis: {
      title: {
        text: "CoordinateX_Iced",
        font: {
          size: 22
        }
      },
      range: ranges.x,
      tickmode: "linear",
      dtick: 0.05,
      tickformat: ".3f",
      ticks: "outside",
      ticklen: 8,
      tickwidth: 2,
      showline: true,
      linewidth: 2,
      mirror: true,
      showgrid: true,
      zeroline: false
    },

    yaxis: {
      title: {
        text: "CoordinateY_Iced",
        font: {
          size: 22
        }
      },
      range: ranges.y,
      tickmode: "linear",
      dtick: 0.01,
      tickformat: ".3f",
      ticks: "outside",
      ticklen: 8,
      tickwidth: 2,
      showline: true,
      linewidth: 2,
      mirror: true,
      showgrid: true,
      zeroline: false,
      scaleanchor: "x",
      scaleratio: 1
    },

    legend: {
      orientation: "v",
      x: 1.02,
      xanchor: "left",
      y: 1.0,
      yanchor: "top"
    },

    margin: {
      l: 90,
      r: 40,
      t: 80,
      b: 80
    },

    plot_bgcolor: "white",
    paper_bgcolor: "white"
  };
}

function buildConfig() {
  return {
    responsive: true,
    displaylogo: false,
    toImageButtonOptions: {
      format: "png",
      filename: "IPW1_Case_241_ice_shape",
      height: 900,
      width: 1200,
      scale: 3
    },
    modeBarButtonsToAdd: [
      {
        name: "Download SVG",
        icon: Plotly.Icons.camera,
        click: graphDiv => Plotly.downloadImage(graphDiv, {
          format: "svg",
          filename: "IPW1_Case_241_ice_shape",
          height: 900,
          width: 1200
        })
      }
    ]
  };
}

async function initialize() {
  const status = document.getElementById("status");

  try {
    const traces = (await Promise.all(datasets.map(dataset => loadDataset(dataset)))).flat();
    const ranges = computeRanges(traces);

    await Plotly.newPlot("plot", traces, buildLayout(ranges), buildConfig());

    status.textContent = "Data loaded successfully.";
  } catch (error) {
    status.textContent = `Error: ${error.message}`;
    console.error(error);
  }
}

initialize();