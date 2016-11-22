#!/bin/bash
montage -geometry -0-0 -crop 315x260+15+60 -resize 100% -tile 3x2 Bool.png ScalarArea.png AreaNormal.png QuadForm.png VertexAngularDefect.png EdgeAngularDefect.png featureExamples.png
montage -geometry -0-0 -crop 315x260+15+60 -resize 95% -tile 2x1 solid.png wireframe.png original.png
