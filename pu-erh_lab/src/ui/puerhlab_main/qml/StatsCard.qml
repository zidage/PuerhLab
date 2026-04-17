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

                        height: 26
                        radius: 13
                        implicitWidth: chipRow.implicitWidth + 20
                        color: isSelected
                            ? Qt.rgba(card.accentColor.r, card.accentColor.g, card.accentColor.b, 0.35)
                            : Qt.rgba(card.accentColor.r, card.accentColor.g, card.accentColor.b, 0.12)

                        MouseArea {
                            anchors.fill: parent
                            cursorShape: Qt.PointingHandCursor
                            onClicked: card.barClicked(entryLabel)
                        }

                        Row {
                            id: chipRow
                            anchors.centerIn: parent
                            spacing: 5

                            Label {
                                text: entryLabel
                                color: isSelected ? appTheme.textColor : appTheme.textMutedColor
                                font.family: appTheme.uiFontFamily
                                font.pixelSize: 11
                                font.weight: isSelected ? 600 : 400
                                anchors.verticalCenter: parent.verticalCenter
                            }

                            Rectangle {
                                anchors.verticalCenter: parent.verticalCenter
                                width: countBadge.implicitWidth + 8
                                height: 16
                                radius: 8
                                color: Qt.rgba(card.accentColor.r, card.accentColor.g, card.accentColor.b, 0.25)

                                Label {
                                    id: countBadge
                                    anchors.centerIn: parent
                                    text: entryCount
                                    color: appTheme.textColor
                                    font.family: appTheme.dataFontFamily
                                    font.pixelSize: 10
                                    font.weight: 600
                                }
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
                    implicitHeight: 22

                    readonly property var entry: card.model[index]
                    readonly property string entryLabel: entry ? String(entry.label) : ""
                    readonly property int entryCount: entry ? Number(entry.count) : 0
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
                            opacity: parent.containsMouse ? 0.04 : 0
                        }
                    }

                    RowLayout {
                        anchors.fill: parent
                        spacing: 8

                        Rectangle {
                            width: 6
                            height: 6
                            radius: 3
                            color: card.accentColor
                            opacity: isSelected ? 1.0 : 0.65
                            Layout.alignment: Qt.AlignVCenter
                        }

                        Label {
                            Layout.fillWidth: true
                            text: entryLabel
                            color: isSelected ? appTheme.textColor
                                              : Qt.rgba(appTheme.textColor.r,
                                                        appTheme.textColor.g,
                                                        appTheme.textColor.b, 0.75)
                            font.family: appTheme.dataFontFamily
                            font.pixelSize: 11
                            font.weight: isSelected ? 600 : 400
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
                    RowLayout {
                        Layout.fillWidth: true

                        Label {
                            text: group ? group.year : ""
                            color: isYearSelected ? appTheme.textColor : appTheme.textMutedColor
                            font.family: appTheme.uiFontFamily
                            font.pixelSize: 12
                            font.weight: isYearSelected ? 700 : 500
                        }

                        Item { Layout.fillWidth: true }

                        Label {
                            text: group ? group.total : 0
                            color: appTheme.textMutedColor
                            font.family: appTheme.dataFontFamily
                            font.pixelSize: 11
                            font.weight: 500
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

                                    height: 22
                                    radius: 11
                                    implicitWidth: pillRow.implicitWidth + 14
                                    color: isSelected
                                        ? Qt.rgba(card.accentColor.r, card.accentColor.g, card.accentColor.b, 0.38)
                                        : Qt.rgba(card.accentColor.r, card.accentColor.g, card.accentColor.b, 0.10)

                                    MouseArea {
                                        anchors.fill: parent
                                        cursorShape: Qt.PointingHandCursor
                                        onClicked: card.barClicked(pill ? pill.label : "")
                                    }

                                    Row {
                                        id: pillRow
                                        anchors.centerIn: parent
                                        spacing: 4

                                        Label {
                                            text: pill ? card.monthAbbr(pill.month) : ""
                                            color: isSelected ? appTheme.textColor : appTheme.textMutedColor
                                            font.family: appTheme.uiFontFamily
                                            font.pixelSize: 10
                                            font.weight: isSelected ? 600 : 400
                                            anchors.verticalCenter: parent.verticalCenter
                                        }

                                        Label {
                                            text: pill ? pill.count : ""
                                            color: isSelected
                                                ? Qt.rgba(card.accentColor.r, card.accentColor.g, card.accentColor.b, 0.9)
                                                : appTheme.textMutedColor
                                            font.family: appTheme.dataFontFamily
                                            font.pixelSize: 10
                                            font.weight: 500
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
