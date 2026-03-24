Option Explicit

Sub RunSimulation()
    ' Settings
    Const NUM_SECURITIES As Integer = 10
    Const DAYS As Integer = 1260 ' 5 years (approx 252 days/year)
    Const INITIAL_PRICE As Double = 100
    Const INV_CAPITAL_TOTAL As Double = 3000 ' Total per security

    Dim wb As Workbook
    Dim wsDash As Worksheet, wsData As Worksheet

    Set wb = ThisWorkbook
    Set wsDash = wb.Sheets("Dashboard")
    Set wsData = wb.Sheets("Simulation Data")

    ' Clear old data
    wsData.Rows("2:" & wsData.Rows.Count).ClearContents
    wsDash.Range("A7:M" & wsDash.Rows.Count).ClearContents

    Dim drifts(1 To NUM_SECURITIES) As Double
    Dim vols(1 To NUM_SECURITIES) As Double
    Dim prices(1 To DAYS, 1 To NUM_SECURITIES) As Double

    Dim i As Integer, d As Integer
    Dim dt As Double
    dt = 1# / 252# ' Daily step in years

    ' 1. Generate random drifts (2% to 10%) and vols (10% to 40%)
    Randomize
    For i = 1 To NUM_SECURITIES
        drifts(i) = 0.02 + Rnd() * 0.08
        vols(i) = 0.1 + Rnd() * 0.3
    Next i

    ' 2. Simulate prices using Geometric Brownian Motion
    For i = 1 To NUM_SECURITIES
        prices(1, i) = INITIAL_PRICE
        For d = 2 To DAYS
            Dim z As Double
            ' Generate standard normal random variable using Box-Muller
            Dim u1 As Double, u2 As Double
            u1 = Rnd()
            If u1 = 0 Then u1 = 0.00000001
            u2 = Rnd()
            z = Sqr(-2 * Log(u1)) * Cos(2 * WorksheetFunction.Pi() * u2)

            ' GBM formula
            prices(d, i) = prices(d - 1, i) * Exp((drifts(i) - 0.5 * vols(i) ^ 2) * dt + vols(i) * Sqr(dt) * z)
        Next d
    Next i

    ' Write prices to Simulation Data
    Dim priceArr() As Variant
    ReDim priceArr(1 To DAYS, 1 To NUM_SECURITIES + 1)
    For d = 1 To DAYS
        priceArr(d, 1) = d
        For i = 1 To NUM_SECURITIES
            priceArr(d, i + 1) = prices(d, i)
        Next i
    Next d
    wsData.Range("A2").Resize(DAYS, NUM_SECURITIES + 1).Value = priceArr

    ' 3. Execute Trade Logic
    Dim totalProfitInv1 As Double
    Dim totalProfitInv2 As Double

    Dim dashArr() As Variant
    ReDim dashArr(1 To NUM_SECURITIES, 1 To 13)

    For i = 1 To NUM_SECURITIES
        ' Dash info
        dashArr(i, 1) = "Security " & i
        dashArr(i, 2) = drifts(i)
        dashArr(i, 3) = vols(i)

        ' INVESTOR 1: random start date
        Dim d1 As Integer
        ' pick a random date between 1 and DAYS - 1 (leave at least 1 day for holding)
        d1 = Int((DAYS - 1 - 1 + 1) * Rnd + 1)

        Dim price1 As Double
        price1 = prices(d1, i)

        Dim shares1 As Double
        shares1 = INV_CAPITAL_TOTAL / price1

        Dim finalVal1 As Double
        finalVal1 = shares1 * prices(DAYS, i)
        Dim profit1 As Double
        profit1 = finalVal1 - INV_CAPITAL_TOTAL

        dashArr(i, 4) = profit1 ' Inv 1 Profit
        dashArr(i, 6) = d1      ' Inv 1 Trade Day
        dashArr(i, 7) = price1  ' Inv 1 Trade Price

        totalProfitInv1 = totalProfitInv1 + profit1

        ' INVESTOR 2: scale-in
        Dim shares2 As Double
        Dim investedCap2 As Double
        shares2 = 0
        investedCap2 = 0

        ' Trade 1: 1/3 on d1
        Dim capThird As Double
        capThird = INV_CAPITAL_TOTAL / 3#

        shares2 = shares2 + capThird / price1
        investedCap2 = investedCap2 + capThird

        dashArr(i, 8) = d1      ' Inv 2 Trade 1 Day
        dashArr(i, 9) = price1  ' Inv 2 Trade 1 Price

        ' Trade 2: Random date > d1 where price < price1
        Dim eligibleDays2() As Integer
        Dim count2 As Integer
        count2 = 0

        Dim j As Integer
        For j = d1 + 1 To DAYS - 1
            If prices(j, i) < price1 Then
                count2 = count2 + 1
                ReDim Preserve eligibleDays2(1 To count2)
                eligibleDays2(count2) = j
            End If
        Next j

        Dim d2 As Integer
        d2 = 0
        Dim price2 As Double
        price2 = 0

        If count2 > 0 Then
            Dim r2 As Integer
            r2 = Int((count2 - 1 + 1) * Rnd + 1)
            d2 = eligibleDays2(r2)
            price2 = prices(d2, i)

            shares2 = shares2 + capThird / price2
            investedCap2 = investedCap2 + capThird

            dashArr(i, 10) = d2
            dashArr(i, 11) = price2
        Else
            dashArr(i, 10) = "N/A"
            dashArr(i, 11) = "N/A"
        End If

        ' Trade 3: Random date > d2 (if d2 exists) where price < price1
        Dim eligibleDays3() As Integer
        Dim count3 As Integer
        count3 = 0
        Dim d3 As Integer
        d3 = 0
        Dim price3 As Double
        price3 = 0

        If d2 > 0 Then
            For j = d2 + 1 To DAYS - 1
                If prices(j, i) < price1 Then
                    count3 = count3 + 1
                    ReDim Preserve eligibleDays3(1 To count3)
                    eligibleDays3(count3) = j
                End If
            Next j

            If count3 > 0 Then
                Dim r3 As Integer
                r3 = Int((count3 - 1 + 1) * Rnd + 1)
                d3 = eligibleDays3(r3)
                price3 = prices(d3, i)

                shares2 = shares2 + capThird / price3
                investedCap2 = investedCap2 + capThird

                dashArr(i, 12) = d3
                dashArr(i, 13) = price3
            Else
                dashArr(i, 12) = "N/A"
                dashArr(i, 13) = "N/A"
            End If
        Else
            dashArr(i, 12) = "N/A"
            dashArr(i, 13) = "N/A"
        End If

        ' Inv 2 Final Profit
        Dim finalVal2 As Double
        ' They hold uninvested capital in cash (0% return), which just adds to final value or we only compute profit on invested capital
        ' Since they started with 3000, final total = shares2 * final_price + (INV_CAPITAL_TOTAL - investedCap2)
        ' Profit = final total - 3000
        finalVal2 = (shares2 * prices(DAYS, i)) + (INV_CAPITAL_TOTAL - investedCap2)
        Dim profit2 As Double
        profit2 = finalVal2 - INV_CAPITAL_TOTAL

        dashArr(i, 5) = profit2 ' Inv 2 Profit
        totalProfitInv1 = totalProfitInv1 + profit1
        totalProfitInv2 = totalProfitInv2 + profit2
    Next i

    ' Output Dashboard Details
    wsDash.Range("A7").Resize(NUM_SECURITIES, 13).Value = dashArr

    ' Format percentages
    wsDash.Range("B7:C" & 6 + NUM_SECURITIES).NumberFormat = "0.00%"
    ' Format money
    wsDash.Range("D7:E" & 6 + NUM_SECURITIES).NumberFormat = "$#,##0.00"
    wsDash.Range("G7:G" & 6 + NUM_SECURITIES).NumberFormat = "$#,##0.00"
    wsDash.Range("I7:I" & 6 + NUM_SECURITIES).NumberFormat = "$#,##0.00"
    wsDash.Range("K7:K" & 6 + NUM_SECURITIES).NumberFormat = "$#,##0.00"
    wsDash.Range("M7:M" & 6 + NUM_SECURITIES).NumberFormat = "$#,##0.00"

    ' Output Totals
    ' Wait, we double counted totalProfitInv1 in the loop (totalProfitInv1 = totalProfitInv1 + profit1 was called twice)
    ' Let me fix the sum recalculation here safely
    totalProfitInv1 = 0
    totalProfitInv2 = 0
    For i = 1 To NUM_SECURITIES
        totalProfitInv1 = totalProfitInv1 + dashArr(i, 4)
        totalProfitInv2 = totalProfitInv2 + dashArr(i, 5)
    Next i

    wsDash.Range("D3").Value = totalProfitInv1
    wsDash.Range("D4").Value = totalProfitInv2
    wsDash.Range("D3:D4").NumberFormat = "$#,##0.00"

    MsgBox "Simulation Complete!", vbInformation
End Sub
