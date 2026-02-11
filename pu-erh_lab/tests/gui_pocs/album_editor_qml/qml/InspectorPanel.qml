import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ScrollView {
    id: root
    contentWidth: availableWidth

    property bool drawerOpen: true
    readonly property int filterFieldFocalLength: 1
    readonly property int filterFieldAperture: 2
    readonly property int filterFieldIso: 3
    readonly property int filterFieldRating: 9
    readonly property int compareOpBetween: 10

    function idxByValue(options, v) {
        for (let i = 0; i < options.length; ++i) {
            if (Number(options[i].value) === Number(v)) {
                return i;
            }
        }
        return 0;
    }

    ColumnLayout {
        width: root.availableWidth
        spacing: 10

        Rectangle {
            Layout.fillWidth: true
            implicitHeight: filterCol.implicitHeight + 20
            radius: 0
            color: "transparent"
            border.width: 0
            ColumnLayout {
                id: filterCol
                anchors.fill: parent
                anchors.margins: 4
                RowLayout {
                    Layout.fillWidth: true
                    Label {
                        text: "Filters"
                        color: "#E3DFDB"
                        font.pixelSize: 15
                        font.weight: 600
                    }
                    Item {
                        Layout.fillWidth: true
                    }
                    Button {
                        text: root.drawerOpen ? "Collapse" : "Expand"
                        onClicked: root.drawerOpen = !root.drawerOpen
                    }
                }

                ColumnLayout {
                    visible: root.drawerOpen
                    Layout.fillWidth: true
                    Label {
                        Layout.fillWidth: true
                        text: "Image metadata rules"
                        color: "#666666"
                        font.pixelSize: 11
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        Label {
                            text: "Rules"
                            color: "#E3DFDB"
                        }
                        Item {
                            Layout.fillWidth: true
                        }
                        ComboBox {
                            id: joinCombo
                            model: [
                                {
                                    text: "Match all rules",
                                    value: 0
                                },
                                {
                                    text: "Match any rule",
                                    value: 1
                                }
                            ]
                            textRole: "text"
                            valueRole: "value"

                            font.pixelSize: 9

                            delegate: ItemDelegate {
                                width: joinCombo.width
                                contentItem: Text {
                                    text: modelData[joinCombo.textRole]

                                    font.pixelSize: 9
                                    color: "#ffffff"
                                    verticalAlignment: Text.AlignVCenter
                                    elide: Text.ElideRight
                                }
                                highlighted: joinCombo.highlightedIndex === index
                            }
                        }
                    }

                    ListView {
                        id: rules
                        Layout.fillWidth: true
                        Layout.preferredHeight: 280
                        clip: true
                        spacing: 6
                        model: albumBackend.filterRules
                        delegate: Rectangle {
                            required property int index
                            required property int fieldValue
                            required property int opValue
                            required property string valueText
                            required property string value2Text
                            required property bool showSecondValue
                            required property string placeholder
                            required property var opOptions
                            readonly property int ruleRow: index
                            readonly property bool isoRangeField: fieldValue === root.filterFieldIso
                            readonly property bool focalRangeField: fieldValue === root.filterFieldFocalLength
                            readonly property bool apertureRangeField: fieldValue === root.filterFieldAperture
                            readonly property bool ratingRangeField: fieldValue === root.filterFieldRating
                            readonly property bool rangeSliderCapable: isoRangeField || focalRangeField || apertureRangeField || ratingRangeField
                            readonly property bool usesRangeSlider: rangeSliderCapable
                            readonly property bool betweenOp: opValue === root.compareOpBetween
                            readonly property real rangeMin: isoRangeField ? 50 : (focalRangeField ? 8 : (apertureRangeField ? 0.7 : 0))
                            readonly property real rangeMax: isoRangeField ? 25600 : (focalRangeField ? 1200 : (apertureRangeField ? 22 : 5))
                            readonly property real rangeStep: isoRangeField ? 50 : (apertureRangeField ? 0.1 : 1)
                            readonly property int rangeDecimals: apertureRangeField ? 1 : 0
                            width: rules.width
                            height: betweenOp ? (usesRangeSlider ? 136 : 66) : 66
                            radius: 6
                            color: "#1F1F1F"
                            border.width: 0

                            function clampRangeValue(value) {
                                return Math.max(rangeMin, Math.min(rangeMax, value));
                            }

                            function parsedRangeValue(text, fallbackValue) {
                                const parsed = Number(text);
                                if (!isFinite(parsed)) {
                                    return clampRangeValue(fallbackValue);
                                }
                                return clampRangeValue(parsed);
                            }

                            function rangeText(value) {
                                const stepped = clampRangeValue(Math.round(value / rangeStep) * rangeStep);
                                if (rangeDecimals === 0) {
                                    return String(Math.round(stepped));
                                }
                                return Number(stepped).toFixed(rangeDecimals);
                            }

                            function defaultSingleValue() {
                                if (isoRangeField) {
                                    return 400;
                                }
                                if (focalRangeField) {
                                    return 50;
                                }
                                if (apertureRangeField) {
                                    return 2.8;
                                }
                                if (ratingRangeField) {
                                    return 3;
                                }
                                return rangeMin;
                            }

                            function defaultLowValue() {
                                if (isoRangeField) {
                                    return 100;
                                }
                                if (focalRangeField) {
                                    return 24;
                                }
                                if (apertureRangeField) {
                                    return 1.4;
                                }
                                if (ratingRangeField) {
                                    return 1;
                                }
                                return rangeMin;
                            }

                            function defaultHighValue() {
                                if (isoRangeField) {
                                    return 6400;
                                }
                                if (focalRangeField) {
                                    return 200;
                                }
                                if (apertureRangeField) {
                                    return 8;
                                }
                                if (ratingRangeField) {
                                    return 5;
                                }
                                return rangeMax;
                            }

                            function syncSingleValue(sliderValue) {
                                albumBackend.setRuleValue(ruleRow, rangeText(sliderValue));
                                if (value2Text.length > 0) {
                                    albumBackend.setRuleValue2(ruleRow, "");
                                }
                            }

                            function initializeRangeRule() {
                                if (!rangeSliderCapable) {
                                    return;
                                }

                                if (betweenOp) {
                                    let low = parsedRangeValue(valueText, defaultLowValue());
                                    let high = parsedRangeValue(value2Text, defaultHighValue());
                                    if (low > high) {
                                        const temp = low;
                                        low = high;
                                        high = temp;
                                    }
                                    const lowText = rangeText(low);
                                    const highText = rangeText(high);
                                    if (valueText !== lowText) {
                                        albumBackend.setRuleValue(ruleRow, lowText);
                                    }
                                    if (value2Text !== highText) {
                                        albumBackend.setRuleValue2(ruleRow, highText);
                                    }
                                } else {
                                    const value = parsedRangeValue(valueText, defaultSingleValue());
                                    const valueAsText = rangeText(value);
                                    if (valueText !== valueAsText) {
                                        albumBackend.setRuleValue(ruleRow, valueAsText);
                                    }
                                    if (value2Text.length > 0) {
                                        albumBackend.setRuleValue2(ruleRow, "");
                                    }
                                }
                            }

                            function syncRangeValues(lowValue, highValue) {
                                let low = lowValue;
                                let high = highValue;
                                if (low > high) {
                                    const temp = low;
                                    low = high;
                                    high = temp;
                                }
                                albumBackend.setRuleValue(ruleRow, rangeText(low));
                                albumBackend.setRuleValue2(ruleRow, rangeText(high));
                            }

                            Component.onCompleted: initializeRangeRule()
                            onFieldValueChanged: initializeRangeRule()
                            onOpValueChanged: initializeRangeRule()

                            Column {
                                anchors.fill: parent
                                anchors.margins: 6
                                spacing: 12
                                Row {
                                    width: parent.width
                                    spacing: 4
                                    ComboBox {
                                        id: fieldOptionsCombo
                                        width: parent.width * 0.43
                                        model: albumBackend.fieldOptions
                                        textRole: "text"
                                        valueRole: "value"
                                        currentIndex: root.idxByValue(model, fieldValue)
                                        onActivated: function () {
                                            albumBackend.setRuleField(ruleRow, currentValue);
                                        }
                                        font.pixelSize: 12
                                        delegate: ItemDelegate {
                                            width: fieldOptionsCombo.width
                                            contentItem: Text {
                                                text: modelData[fieldOptionsCombo.textRole]

                                                font.pixelSize: 12
                                                color: "#ffffff"
                                                verticalAlignment: Text.AlignVCenter
                                                elide: Text.ElideRight
                                            }
                                        }
                                    }
                                    ComboBox {
                                        id: opOptionsCombo
                                        width: parent.width * 0.35
                                        model: opOptions
                                        textRole: "text"
                                        valueRole: "value"
                                        currentIndex: root.idxByValue(opOptions, opValue)
                                        onActivated: function () {
                                            albumBackend.setRuleOp(ruleRow, currentValue);
                                        }
                                        font.pixelSize: 12
                                        delegate: ItemDelegate {
                                            width: opOptionsCombo.width
                                            contentItem: Text {
                                                text: modelData[opOptionsCombo.textRole]

                                                font.pixelSize: 12
                                                color: "#ffffff"
                                                verticalAlignment: Text.AlignVCenter
                                                elide: Text.ElideRight
                                            }
                                        }
                                    }
                                    Button {
                                        width: parent.width * 0.18
                                        text: "X"
                                        onClicked: function () {
                                            albumBackend.removeRule(ruleRow);
                                        }
                                    }
                                }
                                Row {
                                    width: parent.width
                                    spacing: 4
                                    visible: !usesRangeSlider
                                    TextField {
                                        width: showSecondValue ? parent.width * 0.49 : parent.width
                                        text: valueText
                                        placeholderText: placeholder
                                        onTextEdited: function (editedText) {
                                            albumBackend.setRuleValue(ruleRow, editedText);
                                        }
                                    }
                                    TextField {
                                        visible: showSecondValue
                                        width: parent.width * 0.49
                                        text: value2Text
                                        placeholderText: placeholder
                                        onTextEdited: function (editedText) {
                                            albumBackend.setRuleValue2(ruleRow, editedText);
                                        }
                                    }
                                }
                                Column {
                                    width: parent.width
                                    visible: usesRangeSlider
                                    spacing: 4
                                    RangeSlider {
                                        id: rangeSlider
                                        width: parent.width
                                        from: rangeMin
                                        to: rangeMax
                                        stepSize: rangeStep
                                        first.value: parsedRangeValue(valueText, defaultSingleValue())
                                        second.value: betweenOp ? parsedRangeValue(value2Text, defaultHighValue()) : rangeMax
                                        first.onMoved: {
                                            if (betweenOp) {
                                                syncRangeValues(first.value, second.value);
                                            } else {
                                                syncSingleValue(first.value);
                                            }
                                        }
                                        second.onMoved: {
                                            if (betweenOp) {
                                                syncRangeValues(first.value, second.value);
                                            } else {
                                                second.value = rangeMax;
                                            }
                                        }
                                    }
                                    RowLayout {
                                        width: parent.width
                                        Label {
                                            text: rangeText(rangeSlider.first.value)
                                            color: "#E3DFDB"
                                            font.pixelSize: 12
                                        }
                                        Item {
                                            Layout.fillWidth: true
                                        }
                                        Label {
                                            visible: betweenOp
                                            text: rangeText(rangeSlider.second.value)
                                            color: "#E3DFDB"
                                            font.pixelSize: 12
                                        }
                                    }
                                }
                            }
                        }
                    }

                    Button {
                        text: "Add rule"
                        onClicked: albumBackend.addRule()
                    }
                    RowLayout {
                        Layout.fillWidth: true
                        Button {
                            text: "Clear"
                            onClicked: {
                                joinCombo.currentIndex = 0;
                                albumBackend.clearFilters();
                            }
                        }
                        Item {
                            Layout.fillWidth: true
                        }
                        Button {
                            text: "Apply"
                            onClicked: albumBackend.applyFilters(joinCombo.currentValue)
                        }
                    }
                    Label {
                        text: albumBackend.filterInfo
                        color: "#7B7D7C"
                        font.pixelSize: 9
                    }
                    Label {
                        visible: albumBackend.validationError.length > 0
                        text: albumBackend.validationError
                        color: "#E3DFDB"
                        wrapMode: Text.WordWrap
                        font.pixelSize: 9
                    }
                    Label {
                        visible: albumBackend.sqlPreview.length > 0
                        text: albumBackend.sqlPreview
                        wrapMode: Text.WrapAnywhere
                        color: "#E3DFDB"
                        font.family: "Consolas"
                        font.pixelSize: 11
                    }
                }
            }
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 1
            color: "#363636"
        }

        Rectangle {
            Layout.fillWidth: true
            implicitHeight: editorCol.implicitHeight + 20
            radius: 0
            color: "transparent"
            border.width: 0
            ColumnLayout {
                id: editorCol
                anchors.fill: parent
                anchors.margins: 4
                spacing: 8

                RowLayout {
                    Layout.fillWidth: true
                    Label {
                        text: "Editor"
                        color: "#E3DFDB"
                        font.pixelSize: 15
                        font.weight: 600
                    }
                    Item {
                        Layout.fillWidth: true
                    }
                }

                Label {
                    Layout.fillWidth: true
                    text: "The full OpenGL editor opens in a separate dialog window."
                    color: "#666666"
                    font.pixelSize: 11
                    wrapMode: Text.WordWrap
                }

                Label {
                    Layout.fillWidth: true
                    text: albumBackend.editorStatus
                    color: "#E3DFDB"
                    font.pixelSize: 12
                    wrapMode: Text.WordWrap
                }

                Label {
                    visible: albumBackend.editorTitle.length > 0
                    Layout.fillWidth: true
                    text: albumBackend.editorTitle
                    color: "#E3DFDB"
                    font.pixelSize: 12
                    elide: Text.ElideRight
                }

                Label {
                    Layout.fillWidth: true
                    text: albumBackend.editorActive ? "Editor window is open. Close that window to save edits." : "Click Edit on a thumbnail to open the full editor dialog."
                    color: "#666666"
                    font.pixelSize: 11
                    wrapMode: Text.WordWrap
                }
            }
        }
    }
}
