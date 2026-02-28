import QtQuick

/**
 * Apple Watch–style activity ring that fills clockwise from 12 o'clock.
 *
 * Properties
 *   progress   : 0.0 – 1.0  (fraction completed)
 *   ringWidth  : stroke thickness in px
 *   trackColor : background ring colour
 *   fillColor  : foreground arc colour
 */
Item {
    id: ring
    width: 160
    height: 160

    property real progress: 0.0
    property real ringWidth: 14
    property color trackColor: "#333333"
    property color fillColor: "#FCC704"

    // Smoothly animate the arc whenever progress changes
    Behavior on progress {
        NumberAnimation { duration: 350; easing.type: Easing.OutCubic }
    }

    Canvas {
        id: canvas
        anchors.fill: parent
        antialiasing: true

        onPaint: {
            var ctx = getContext("2d");
            ctx.reset();

            var cx     = width  / 2;
            var cy     = height / 2;
            var radius = Math.min(cx, cy) - ring.ringWidth / 2 - 2;
            var startAngle = -Math.PI / 2;               // 12 o'clock
            var endAngle   = startAngle + 2 * Math.PI * Math.min(ring.progress, 1.0);

            // ── track (background) ──
            ctx.beginPath();
            ctx.arc(cx, cy, radius, 0, 2 * Math.PI);
            ctx.lineWidth = ring.ringWidth;
            ctx.strokeStyle = ring.trackColor;
            ctx.lineCap = "round";
            ctx.stroke();

            // ── filled arc ──
            if (ring.progress > 0.001) {
                ctx.beginPath();
                ctx.arc(cx, cy, radius, startAngle, endAngle);
                ctx.lineWidth = ring.ringWidth;
                ctx.strokeStyle = ring.fillColor;
                ctx.lineCap = "round";
                ctx.stroke();
            }

            // ── end-cap glow dot ──
            if (ring.progress > 0.02 && ring.progress < 1.0) {
                var dotX = cx + radius * Math.cos(endAngle);
                var dotY = cy + radius * Math.sin(endAngle);
                ctx.beginPath();
                ctx.arc(dotX, dotY, ring.ringWidth / 2 + 2, 0, 2 * Math.PI);
                ctx.fillStyle = Qt.rgba(ring.fillColor.r, ring.fillColor.g, ring.fillColor.b, 0.35);
                ctx.fill();
            }
        }
    }

    // Repaint whenever progress changes
    onProgressChanged: canvas.requestPaint()
    Component.onCompleted: canvas.requestPaint()
    onWidthChanged: canvas.requestPaint()
    onHeightChanged: canvas.requestPaint()
}
