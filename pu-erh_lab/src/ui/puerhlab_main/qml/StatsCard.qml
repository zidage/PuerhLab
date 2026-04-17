import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    id: card
    implicitHeight: cardCol.implicitHeight

    property string title: ""
    property color accentColor: appTheme.accentColor
    property var model: []
    property bool expanded: true
    readonly property int previewLimit: 10
    property string selectedLabel: ""
    property string displayMode: "bars"
    signal barClicked(string label)

    readonly property var groupedItems: {
        if (!model) return [];
        const groups = {};
        const order = [];
        for (let i = 0; i < model.length; i++) {
            const entry = model[i];
            const label = String(entry.label);
            const m = label.match(/^(\d{4})(-(\d{2}))?/);
            const year = m ? m[1] : label;
            if (!groups[year]) {
                groups[year] = { year: year, total: 0, items: [] };
                order.push(year);
            }
            groups[year].total += Number(entry.count);
            if (m && m[3]) {
                groups[year].items.push({ label: label, month: m[3], count: Number(entry.count) });
            }
        }
        return order.map(y => groups[y]);
    }

    function monthAbbr(mm) {
        const names = ["Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"];
        const idx = parseInt(mm) - 1;
        return (idx >= 0 && idx < 12) ? names[idx] : mm;
    }

    function withAlpha(color, alpha) {
        return Qt.rgba(color.r, color.g, color.b, alpha);
    }

    function lensDotColor(index, isSelected) {
        if (isSelected || index === 0) return card.accentColor;
        if (index === 1) return appTheme.accentSecondaryColor;
        if (index === 2) return card.withAlpha(appTheme.textMutedColor, 0.55);
        return card.withAlpha(appTheme.textMutedColor, 0.75);
    }

    readonly property int totalItems: model ? model.length : 0
    readonly property bool hasOverflow: totalItems > previewLimit
    property bool showAll: false
    readonly property int visibleCount: showAll ? totalItems : Math.min(totalItems, previewLimit)

    function maxCount() {
        let m = 1;
        if (!model) return m;
        const n = Math.min(model.length, visibleCount);
        for (let i = 0; i < n; ++i) {
            const c = Number(model[i].count);
            if (c > m) m = c;
        }
        return m;
    }

    ColumnLayout {
        id: cardCol
        anchors.left: parent.left
        anchors.right: parent.right
        spacing: 8

        RowLayout {
            Layout.fillWidth: true

            Label {
                text: card.title.toUpperCase()
                color: appTheme.textMutedColor
                font.pixelSize: 10
                font.weight: 700
                font.letterSpacing: 1.6
            }
            Item { Layout.fillWidth: true }
            Label {
                text: card.expanded ? "▲" : "▼"
                color: appTheme.textMutedColor
                font.pixelSize: 9
                MouseArea {
                    anchors.fill: parent
                    cursorShape: Qt.PointingHandCursor
                    onClicked: card.expanded = !card.expanded
                }
            }
        }

        // --- bars ---
        ColumnLayout {
            visible: card.displayMode === "bars" && card.expanded
            Layout.fillWidth: true
            spacing: 3

            Repeater {
                model: card.visibleCount
                delegate: Item {
                    required property int index
                    Layout.fillWidth: true
                    implicitHeight: 24

                    readonly property var entry: card.model[index]
                    readonly property string entryLabel: entry ? String(entry.label) : ""
                    readonly property int entryCount: entry ? Number(entry.count) : 0
                    readonly property real fraction: entryCount / card.maxCount()
                    readonly property bool isSelected: entryLabel === card.selectedLabel
                                                      && card.selectedLabel !== ""

                    MouseArea {
                        anchors.fill: parent
                        cursorShape: Qt.PointingHandCursor
                        hoverEnabled: true
                        onClicked: card.barClicked(entryLabel)

                        Rectangle {
                            anchors.fill: parent
                            radius: 3
                            color: "#FFFFFF"
                            opacity: parent.containsMouse && !isSelected ? 0.04 : 0
                        }
                    }

                    Rectangle {
                        anchors.left: parent.left
                        anchors.verticalCenter: parent.verticalCenter
                        width: parent.width * fraction
                        height: 20
                        radius: 3
                        color: card.accentColor
                        opacity: isSelected ? 0.55 : 0.18
                    }

                    Rectangle {
                        visible: isSelected
                        anchors.left: parent.left
                        anchors.verticalCenter: parent.verticalCenter
                        width: 3
                        height: 16
                        radius: 1.5
                        color: card.accentColor
                    }

                    RowLayout {
                        anchors.fill: parent
                        anchors.leftMargin: 6
                        anchors.rightMargin: 6

                        Label {
                            Layout.fillWidth: true
                            text: entryLabel
                            color: appTheme.textColor
                            font.family: appTheme.dataFontFamily
                            font.pixelSize: 11
                            font.weight: 400
                            elide: Text.ElideRight
                        }

                        Label {
                            text: entryCount
                            color: appTheme.textMutedColor
                            font.family: appTheme.dataFontFamily
                            font.pixelSize: 11
                            font.weight: 500
                        }
                    }
                }
            }

            Label {
                visible: card.totalItems === 0
                text: qsTr("No data available")
                color: appTheme.textMutedColor
                font.pixelSize: 11
                font.italic: true
                Layout.alignment: Qt.AlignHCenter
                Layout.topMargin: 4
            }
        }

        // --- chips ---
        Item {
            visible: card.displayMode === "chips" && card.expanded
            Layout.fillWidth: true
            implicitHeight: chipsFlow.implicitHeight

            Flow {
                id: chipsFlow
                width: parent.width
                spacing: 6

                Repeater {
                    model: card.visibleCount
                    delegate: Rectangle {
                        required property int index
                        readonly property var entry: card.model[index]
                        readonly property string entryLabel: entry ? String(entry.label) : ""
                        readonly property int entryCount: entry ? Number(entry.count) : 0
                        readonly property bool isSelected: entryLabel === card.selectedLabel
                                                          && card.selectedLabel !== ""

                        id: chip
                        height: 36
                        radius: 18
                        implicitWidth: chipRow.implicitWidth + 26
                        color: isSelected
                            ? card.withAlpha(appTheme.selectedTintColor, 0.26)
                            : chipHitArea.containsMouse
                                ? card.withAlpha(appTheme.hoverColor, 0.92)
                                : card.withAlpha(appTheme.bgBaseColor, 0.82)
                        border.width: 1
                        border.color: isSelected
                            ? card.withAlpha(card.accentColor, 0.55)
                            : card.withAlpha(appTheme.glassStrokeColor, 0.7)

                        MouseArea {
                            id: chipHitArea
                            anchors.fill: parent
                            cursorShape: Qt.PointingHandCursor
                            hoverEnabled: true
                            onClicked: card.barClicked(entryLabel)
                        }

                        Row {
                            id: chipRow
                            anchors.centerIn: parent
                            spacing: 4

                            Label {
                                text: entryLabel
                                color: isSelected ? card.accentColor : appTheme.textColor
                                font.family: appTheme.uiFontFamily
                                font.pixelSize: 11
                                font.weight: isSelected ? 700 : 600
                                anchors.verticalCenter: parent.verticalCenter
                            }

                            Label {
                                text: "(" + entryCount + ")"
                                color: isSelected
                                    ? card.withAlpha(card.accentColor, 0.92)
                                    : appTheme.textMutedColor
                                font.family: appTheme.dataFontFamily
                                font.pixelSize: 11
                                font.weight: 600
                                anchors.verticalCenter: parent.verticalCenter
                            }
                        }
                    }
                }

                Label {
                    visible: card.totalItems === 0
                    text: qsTr("No data available")
                    color: appTheme.textMutedColor
                    font.pixelSize: 11
                    font.italic: true
                }
            }
        }

        // --- dots ---
        ColumnLayout {
            visible: card.displayMode === "dots" && card.expanded
            Layout.fillWidth: true
            spacing: 6

            Repeater {
                model: card.visibleCount
                delegate: Item {
                    required property int index
                    Layout.fillWidth: true
                    implicitHeight: 42

                    readonly property var entry: card.model[index]
                    readonly property string entryLabel: entry ? String(entry.label) : ""
                    readonly property int entryCount: entry ? Number(entry.count) : 0
                    readonly property bool isSelected: entryLabel === card.selectedLabel
                                                      && card.selectedLabel !== ""

                    Rectangle {
                        anchors.fill: parent
                        radius: 6
                        color: isSelected
                            ? card.withAlpha(appTheme.bgBaseColor, 0.92)
                            : dotHitArea.containsMouse
                                ? card.withAlpha(appTheme.hoverColor, 0.86)
                                : "transparent"
                        border.width: isSelected ? 1 : 0
                        border.color: card.withAlpha(card.accentColor, 0.42)
                    }

                    MouseArea {
                        id: dotHitArea
                        anchors.fill: parent
                        cursorShape: Qt.PointingHandCursor
                        hoverEnabled: true
                        onClicked: card.barClicked(entryLabel)
                    }

                    RowLayout {
                        anchors.fill: parent
                        anchors.leftMargin: 12
                        anchors.rightMargin: 12
                        spacing: 10

                        Rectangle {
                            implicitWidth: 9
                            implicitHeight: 9
                            radius: 4.5
                            color: card.lensDotColor(index, isSelected)
                            Layout.alignment: Qt.AlignVCenter
                        }

                        Label {
                            Layout.fillWidth: true
                            text: entryLabel
                            color: isSelected ? appTheme.textColor
                                              : card.withAlpha(appTheme.textColor, 0.78)
                            font.family: appTheme.dataFontFamily
                            font.pixelSize: 12
                            font.weight: isSelected ? 600 : 500
                            elide: Text.ElideRight
                        }

                        Label {
                            text: entryCount
                            color: isSelected
                                ? card.withAlpha(appTheme.textColor, 0.82)
                                : appTheme.textMutedColor
                            font.family: appTheme.dataFontFamily
                            font.pixelSize: 12
                            font.weight: 500
                        }
                    }
                }
            }

            Label {
                visible: card.totalItems === 0
                text: qsTr("No data available")
                color: appTheme.textMutedColor
                font.pixelSize: 11
                font.italic: true
                Layout.alignment: Qt.AlignHCenter
                Layout.topMargin: 4
            }
        }

        // --- grouped (year header + month pills) ---
        ColumnLayout {
            visible: card.displayMode === "grouped" && card.expanded
            Layout.fillWidth: true
            spacing: 12

            Repeater {
                model: card.groupedItems.length

                delegate: ColumnLayout {
                    required property int index
                    Layout.fillWidth: true
                    spacing: 6

                    readonly property var group: card.groupedItems[index]
                    readonly property bool isYearSelected: group
                        ? group.items.some(it => it.label === card.selectedLabel)
                        : false

                    // Year row
                    Rectangle {
                        Layout.fillWidth: true
                        implicitHeight: 38
                        radius: 18
                        color: isYearSelected
                            ? card.withAlpha(appTheme.selectedTintColor, 0.24)
                            : card.withAlpha(appTheme.bgBaseColor, 0.82)
                        border.width: 1
                        border.color: isYearSelected
                            ? card.withAlpha(card.accentColor, 0.5)
                            : card.withAlpha(appTheme.glassStrokeColor, 0.7)

                        RowLayout {
                            anchors.fill: parent
                            anchors.leftMargin: 14
                            anchors.rightMargin: 14

                            Label {
                                text: group ? group.year : ""
                                color: isYearSelected ? card.accentColor : appTheme.textColor
                                font.family: appTheme.uiFontFamily
                                font.pixelSize: 12
                                font.weight: isYearSelected ? 700 : 600
                            }

                            Item { Layout.fillWidth: true }

                            Label {
                                text: group ? "(" + group.total + ")" : ""
                                color: isYearSelected
                                    ? card.withAlpha(card.accentColor, 0.92)
                                    : appTheme.textMutedColor
                                font.family: appTheme.dataFontFamily
                                font.pixelSize: 11
                                font.weight: 600
                            }
                        }
                    }

                    // Month pills
                    Item {
                        visible: group && group.items.length > 0
                        Layout.fillWidth: true
                        implicitHeight: monthFlow.implicitHeight

                        Flow {
                            id: monthFlow
                            width: parent.width
                            spacing: 5

                            Repeater {
                                model: group ? group.items.length : 0
                                delegate: Rectangle {
                                    required property int index
                                    readonly property var pill: group.items[index]
                                    readonly property bool isSelected: pill
                                        ? pill.label === card.selectedLabel
                                        : false

                                    id: monthPill
                                    height: 30
                                    radius: 15
                                    implicitWidth: pillRow.implicitWidth + 20
                                    color: isSelected
                                        ? card.withAlpha(appTheme.selectedTintColor, 0.24)
                                        : monthHitArea.containsMouse
                                            ? card.withAlpha(appTheme.hoverColor, 0.92)
                                            : card.withAlpha(appTheme.bgBaseColor, 0.82)
                                    border.width: 1
                                    border.color: isSelected
                                        ? card.withAlpha(card.accentColor, 0.5)
                                        : card.withAlpha(appTheme.glassStrokeColor, 0.7)

                                    MouseArea {
                                        id: monthHitArea
                                        anchors.fill: parent
                                        cursorShape: Qt.PointingHandCursor
                                        hoverEnabled: true
                                        onClicked: card.barClicked(pill ? pill.label : "")
                                    }

                                    Row {
                                        id: pillRow
                                        anchors.centerIn: parent
                                        spacing: 4

                                        Label {
                                            text: pill ? card.monthAbbr(pill.month) : ""
                                            color: isSelected ? card.accentColor : appTheme.textColor
                                            font.family: appTheme.uiFontFamily
                                            font.pixelSize: 11
                                            font.weight: isSelected ? 700 : 600
                                            anchors.verticalCenter: parent.verticalCenter
                                        }

                                        Label {
                                            text: pill ? "(" + pill.count + ")" : ""
                                            color: isSelected
                                                ? card.withAlpha(card.accentColor, 0.92)
                                                : appTheme.textMutedColor
                                            font.family: appTheme.dataFontFamily
                                            font.pixelSize: 11
                                            font.weight: 600
                                            anchors.verticalCenter: parent.verticalCenter
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            Label {
                visible: card.groupedItems.length === 0
                text: qsTr("No data available")
                color: appTheme.textMutedColor
                font.pixelSize: 11
                font.italic: true
                Layout.alignment: Qt.AlignHCenter
                Layout.topMargin: 4
            }
        }

        Label {
            visible: card.hasOverflow && card.expanded
            text: card.showAll
                ? qsTr("Show less ▲")
                : qsTr("Show all %1 ▼").arg(card.totalItems)
            color: card.accentColor
            font.pixelSize: 10
            font.underline: true
            Layout.alignment: Qt.AlignHCenter
            Layout.topMargin: 2

            MouseArea {
                anchors.fill: parent
                cursorShape: Qt.PointingHandCursor
                onClicked: card.showAll = !card.showAll
            }
        }
    }
}
