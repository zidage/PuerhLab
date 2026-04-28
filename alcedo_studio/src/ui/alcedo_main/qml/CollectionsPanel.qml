import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

ColumnLayout {
    id: panel

    property var backend
    property var theme
    property bool backendInteractive: false
    property var folderRows: []
    property bool sortDescending: false
    property bool draftCollectionVisible: false
    signal importRequested()

    function withAlpha(colorValue, alphaValue) {
        return Qt.rgba(colorValue.r, colorValue.g, colorValue.b, alphaValue)
    }

    function folderSortKey(folder) {
        const fullPath = folder.path ? String(folder.path).toLowerCase() : ""
        const name = folder.name ? String(folder.name).toLowerCase() : ""
        return fullPath.length > 0 ? fullPath : name
    }

    function rebuildFolderRows() {
        const source = backend && backend.folders ? backend.folders : []
        let rootRow = null
        const next = []

        for (let i = 0; i < source.length; ++i) {
            const row = source[i]
            if (!row) {
                continue
            }

            const mapped = {
                folderId: Number(row.folderId),
                name: row.name ? String(row.name) : "",
                depth: Number(row.depth),
                path: row.path ? String(row.path) : "",
                deletable: row.deletable === true
            }

            if (mapped.folderId === 0) {
                rootRow = mapped
                continue
            }

            next.push(mapped)
        }

        next.sort(function(a, b) {
            const left = folderSortKey(a)
            const right = folderSortKey(b)
            if (left === right) {
                return a.folderId - b.folderId
            }
            if (sortDescending) {
                return right < left ? -1 : 1
            }
            return left < right ? -1 : 1
        })

        folderRows = rootRow ? [rootRow].concat(next) : next
    }

    function beginCreateCollection() {
        if (draftCollectionVisible) {
            draftFocusTimer.restart()
            return
        }
        draftCollectionVisible = true
        draftCollectionField.text = ""
        draftFocusTimer.restart()
    }

    function cancelDraftCollection() {
        draftCollectionVisible = false
        draftCollectionField.text = ""
    }

    function commitDraftCollection() {
        if (!draftCollectionVisible) {
            return
        }

        const trimmed = draftCollectionField.text.trim()
        if (trimmed.length === 0) {
            cancelDraftCollection()
            return
        }

        if (!backend) {
            return
        }

        backend.CreateFolder(trimmed)
        cancelDraftCollection()
    }

    readonly property bool hasSelectedCollection: backend && Number(backend.currentFolderId) !== 0

    Layout.preferredWidth: 276
    Layout.minimumWidth: 276
    Layout.maximumWidth: 276
    Layout.fillHeight: true
    spacing: 12

    Component.onCompleted: rebuildFolderRows()
    onSortDescendingChanged: rebuildFolderRows()

    Connections {
        target: backend
        ignoreUnknownSignals: true

        function onFoldersChanged() {
            panel.rebuildFolderRows()
        }

        function onFolderSelectionChanged() {
            panel.rebuildFolderRows()
        }
    }

    Timer {
        id: draftFocusTimer
        interval: 0
        onTriggered: draftCollectionField.forceActiveFocus()
    }

    Rectangle {
        Layout.fillWidth: true
        Layout.fillHeight: true
        radius: theme.panelRadius
        color: Qt.darker(theme.colBgPanel, 1.08)
        border.width: 1
        border.color: panel.withAlpha(theme.colText, 0.05)
        clip: true

        ColumnLayout {
            anchors.fill: parent
            anchors.margins: 14
            spacing: 12

            Label {
                Layout.fillWidth: true
                text: qsTr("Collections")
                color: panel.withAlpha(theme.colText, 0.94)
                font.family: appTheme.headlineFontFamily
                font.pixelSize: 34
                font.weight: 700
            }

            Button {
                id: searchPlaceholderButton
                Layout.fillWidth: true
                Layout.preferredHeight: 38
                Material.foreground: theme.colText
                onClicked: {}

                background: Rectangle {
                    radius: 10
                    color: searchPlaceholderButton.down
                           ? panel.withAlpha(theme.colBgBase, 0.96)
                           : panel.withAlpha(theme.colBgBase, 0.84)
                    border.width: 1
                    border.color: panel.withAlpha(theme.colText, searchPlaceholderButton.hovered ? 0.12 : 0.08)
                }

                contentItem: RowLayout {
                    spacing: 8

                    Image {
                        width: 16
                        height: 16
                        source: "qrc:/panel_icons/search.svg"
                        sourceSize.width: 16
                        sourceSize.height: 16
                        fillMode: Image.PreserveAspectFit
                        mipmap: false
                        smooth: false
                    }

                    Label {
                        text: qsTr("Search")
                        color: panel.withAlpha(theme.colText, 0.76)
                        font.pixelSize: 13
                        font.weight: 500
                    }

                    Item { Layout.fillWidth: true }
                }
            }

            RowLayout {
                Layout.fillWidth: true
                spacing: 8

                Label {
                    text: qsTr("LOCAL FOLDERS")
                    color: panel.withAlpha(theme.colText, 0.5)
                    font.pixelSize: 11
                    font.letterSpacing: 1.2
                    font.weight: 600
                }

                Item { Layout.fillWidth: true }

                Button {
                    id: sortButton
                    Layout.preferredWidth: 28
                    Layout.preferredHeight: 28
                    Material.foreground: theme.colText
                    onClicked: panel.sortDescending = !panel.sortDescending

                    background: Rectangle {
                        radius: width / 2
                        color: sortButton.hovered || panel.sortDescending
                               ? panel.withAlpha(theme.colHover, 0.55)
                               : "transparent"
                    }

                    contentItem: Item {
                        Image {
                            anchors.centerIn: parent
                            width: 16
                            height: 16
                            source: "qrc:/panel_icons/sort.svg"
                            sourceSize.width: 16
                            sourceSize.height: 16
                            fillMode: Image.PreserveAspectFit
                            mipmap: false
                            smooth: false
                            rotation: panel.sortDescending ? 180 : 0

                            Behavior on rotation {
                                NumberAnimation { duration: 160; easing.type: Easing.OutCubic }
                            }
                        }
                    }

                    ToolTip.visible: hovered
                    ToolTip.text: panel.sortDescending ? qsTr("Sorted Z-A") : qsTr("Sorted A-Z")
                }

                Button {
                    id: addCollectionButton
                    Layout.preferredWidth: 28
                    Layout.preferredHeight: 28
                    Material.foreground: theme.colText
                    onClicked: panel.beginCreateCollection()

                    background: Rectangle {
                        radius: width / 2
                        color: addCollectionButton.hovered || draftCollectionVisible
                               ? panel.withAlpha(theme.colHover, 0.55)
                               : "transparent"
                    }

                    contentItem: Item {
                        Image {
                            anchors.centerIn: parent
                            width: 16
                            height: 16
                            source: "qrc:/panel_icons/folder-plus.svg"
                            sourceSize.width: 16
                            sourceSize.height: 16
                            fillMode: Image.PreserveAspectFit
                            mipmap: false
                            smooth: false
                        }
                    }

                    ToolTip.visible: hovered
                    ToolTip.text: qsTr("New collection")
                }
            }

            Item {
                Layout.fillWidth: true
                Layout.preferredHeight: draftCollectionVisible ? 52 : 0
                clip: true

                Behavior on Layout.preferredHeight {
                    NumberAnimation { duration: 180; easing.type: Easing.OutCubic }
                }

                Rectangle {
                    anchors.fill: parent
                    radius: 10
                    color: panel.withAlpha(theme.colHover, 0.32)
                    border.width: 1
                    border.color: panel.withAlpha(theme.colText, 0.08)
                    opacity: draftCollectionVisible ? 1.0 : 0.0

                    Behavior on opacity { NumberAnimation { duration: 120 } }

                    RowLayout {
                        anchors.fill: parent
                        anchors.leftMargin: 10
                        anchors.rightMargin: 10
                        spacing: 10

                        Image {
                            width: 15
                            height: 15
                            source: "qrc:/panel_icons/folder-open.svg"
                            sourceSize.width: 15
                            sourceSize.height: 15
                            fillMode: Image.PreserveAspectFit
                            mipmap: false
                            smooth: false
                        }

                        TextField {
                            id: draftCollectionField
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            placeholderText: qsTr("New collection...")
                            color: theme.colText
                            font.pixelSize: 15
                            selectedTextColor: theme.colBgCanvas
                            selectionColor: panel.withAlpha(theme.colAccentSecondary, 0.6)
                            background: Item {}
                            onAccepted: panel.commitDraftCollection()
                            Keys.onEscapePressed: panel.cancelDraftCollection()
                            onEditingFinished: {
                                if (panel.draftCollectionVisible) {
                                    panel.commitDraftCollection()
                                }
                            }
                        }
                    }
                }
            }

            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: 1
                color: panel.withAlpha(theme.colText, 0.06)
            }

            ListView {
                id: folderList
                Layout.fillWidth: true
                Layout.fillHeight: true
                clip: true
                spacing: 2
                model: panel.folderRows

                ScrollIndicator.vertical: ScrollIndicator {}

                delegate: Item {
                    required property var modelData

                    width: ListView.view.width
                    height: 52

                    readonly property bool selected: modelData.folderId === Number(panel.backend.currentFolderId)

                    Rectangle {
                        anchors.fill: parent
                        anchors.rightMargin: 2
                        radius: 10
                        color: selected
                               ? panel.withAlpha(theme.colHover, 0.54)
                               : folderMouse.containsMouse
                                 ? panel.withAlpha(theme.colHover, 0.28)
                                 : "transparent"
                        border.width: selected ? 1 : 0
                        border.color: selected ? panel.withAlpha(theme.colText, 0.08) : "transparent"

                        Rectangle {
                            anchors.top: parent.top
                            anchors.bottom: parent.bottom
                            anchors.right: parent.right
                            width: selected ? 2 : 0
                            radius: 1
                            color: theme.colAccentSecondary

                            Behavior on width {
                                NumberAnimation { duration: 130 }
                            }
                        }

                        RowLayout {
                            anchors.fill: parent
                            anchors.leftMargin: 10 + modelData.depth * 14
                            anchors.rightMargin: 10
                            spacing: 10

                            Image {
                                width: 15
                                height: 15
                                source: "qrc:/panel_icons/folder-open.svg"
                                sourceSize.width: 15
                                sourceSize.height: 15
                                fillMode: Image.PreserveAspectFit
                                mipmap: false
                                smooth: false
                            }

                            Label {
                                Layout.fillWidth: true
                                text: modelData.name
                                color: selected ? theme.colText : panel.withAlpha(theme.colText, 0.92)
                                font.pixelSize: 15
                                font.weight: selected ? 600 : 400
                                elide: Text.ElideRight
                            }
                        }

                        MouseArea {
                            id: folderMouse
                            anchors.fill: parent
                            hoverEnabled: true
                            cursorShape: Qt.PointingHandCursor
                            onClicked: panel.backend.SelectFolder(modelData.folderId)
                        }
                    }
                }

                Label {
                    anchors.centerIn: parent
                    visible: folderList.count === 0
                    text: qsTr("No collections yet")
                    color: panel.withAlpha(theme.colText, 0.55)
                    font.pixelSize: 13
                }
            }

            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: hasSelectedCollection ? 38 : 0
                radius: 10
                color: hasSelectedCollection ? panel.withAlpha(theme.colDanger, 0.12) : "transparent"
                border.width: hasSelectedCollection ? 1 : 0
                border.color: hasSelectedCollection ? panel.withAlpha(theme.colDanger, 0.22) : "transparent"

                Behavior on Layout.preferredHeight {
                    NumberAnimation { duration: 160; easing.type: Easing.OutCubic }
                }

                Button {
                    anchors.fill: parent
                    visible: hasSelectedCollection
                    enabled: hasSelectedCollection && backendInteractive
                    text: qsTr("Delete collection")
                    Material.foreground: theme.colText
                    onClicked: backend.DeleteFolder(backend.currentFolderId)
                    background: Item {}

                    contentItem: Label {
                        text: parent.text
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                        color: panel.withAlpha(theme.colText, 0.84)
                        font.pixelSize: 12
                        font.weight: 600
                    }
                }
            }
        }
    }

    Button {
        id: importBtn
        Layout.fillWidth: true
        Layout.preferredHeight: 52
        text: qsTr("Import")
        enabled: backendInteractive
        icon.source: "qrc:/panel_icons/import.svg"
        icon.width: 16
        icon.height: 16
        icon.color: theme.colText
        display: AbstractButton.TextBesideIcon
        Material.foreground: theme.colText
        onClicked: panel.importRequested()

        background: Canvas {
            opacity: importBtn.enabled ? 1.0 : 0.5
            property color gradStart: theme.colAccentPrimary
            property color gradEnd: theme.colAccentSecondary
            onGradStartChanged: requestPaint()
            onGradEndChanged: requestPaint()
            onWidthChanged: requestPaint()
            onHeightChanged: requestPaint()
            onPaint: {
                var ctx = getContext("2d")
                ctx.clearRect(0, 0, width, height)
                var r = 8
                ctx.beginPath()
                ctx.moveTo(r, 0)
                ctx.lineTo(width - r, 0)
                ctx.quadraticCurveTo(width, 0, width, r)
                ctx.lineTo(width, height - r)
                ctx.quadraticCurveTo(width, height, width - r, height)
                ctx.lineTo(r, height)
                ctx.quadraticCurveTo(0, height, 0, height - r)
                ctx.lineTo(0, r)
                ctx.quadraticCurveTo(0, 0, r, 0)
                ctx.closePath()
                var grad = ctx.createLinearGradient(0, height, width, 0)
                grad.addColorStop(0.0, Qt.rgba(gradStart.r, gradStart.g, gradStart.b, 1.0))
                grad.addColorStop(1.0, Qt.rgba(gradEnd.r, gradEnd.g, gradEnd.b, 1.0))
                ctx.fillStyle = grad
                ctx.fill()
            }
        }

        scale: importBtn.hovered && enabled ? 1.03 : 1.0
        Behavior on scale { NumberAnimation { duration: 100; easing.type: Easing.OutCubic } }
    }
}
