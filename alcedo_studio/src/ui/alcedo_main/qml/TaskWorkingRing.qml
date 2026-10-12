//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

import QtQuick

// Working indicator of a background task: an open arc that circles a status
// lamp while the task runs. The arc is painted once; a render-thread animator
// turns it. With reduced motion the arc stands still.
Item {
    id: root

    property color color: appTheme.backgroundTaskWorkingColor
    property real strokeWidth: 1.5
    property bool running: true

    implicitWidth: appTheme.spaceLg
    implicitHeight: appTheme.spaceLg
    Accessible.ignored: true

    Canvas {
        id: arc
        anchors.fill: parent
        antialiasing: true

        onPaint: {
            const ctx = getContext("2d")
            ctx.reset()
            const radius = Math.min(width, height) / 2 - root.strokeWidth
            if (radius <= 0)
                return
            ctx.lineWidth = root.strokeWidth
            ctx.lineCap = "round"
            ctx.strokeStyle = Qt.rgba(root.color.r, root.color.g, root.color.b, 0.2)
            ctx.beginPath()
            ctx.arc(width / 2, height / 2, radius, 0, 2 * Math.PI)
            ctx.stroke()
            ctx.strokeStyle = root.color
            ctx.beginPath()
            ctx.arc(width / 2, height / 2, radius, -Math.PI / 2, Math.PI * 0.6)
            ctx.stroke()
        }

        RotationAnimator on rotation {
            running: root.running && root.visible && !appTheme.reduceMotion
            loops: Animation.Infinite
            from: 0
            to: 360
            duration: appTheme.motionFoldOpenMs * 5
        }
    }

    onColorChanged: arc.requestPaint()
    onStrokeWidthChanged: arc.requestPaint()
    onWidthChanged: arc.requestPaint()
    onHeightChanged: arc.requestPaint()
}
