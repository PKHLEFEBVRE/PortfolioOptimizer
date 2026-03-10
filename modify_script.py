import re

with open('original_vba_code.bas', 'r') as f:
    content = f.read()

# Replace strategies arrays
old_array = 'Array("EQUAL WEIGHT", "ERC UNCSTRD", "ER/VOL", "SHARPE", "MIN VAR", "KELLY", "CUSTOM")'
new_array = 'Array("EQUAL WEIGHT", "ERC UNCSTRD", "ER/VOL", "SHARPE", "MIN VAR", "KELLY", "CUSTOM", "MEAN")'
content = content.replace(old_array, new_array)

# Add sumWeights array logic in RunAllSolvers
# Find RunAllSolvers definition
start_runall = content.find('Sub RunAllSolvers()')
if start_runall == -1:
    print("RunAllSolvers not found")
    exit(1)

# Find the loop start inside RunAllSolvers
loop_start_str = 'For i = LBound(strategies) To UBound(strategies)'
loop_start_pos = content.find(loop_start_str, start_runall)

# Code to insert before the loop
insert_before_loop = """
    Dim sumWeights() As Double
    ReDim sumWeights(1 To nAssets)
    Dim wCount As Integer
    wCount = UBound(strategies) - 1 ' All strategies except MEAN

    For m = 1 To nAssets
        sumWeights(m) = 0
    Next m

    """

# Insert sumWeights initialization
content = content[:loop_start_pos] + insert_before_loop + content[loop_start_pos:]

# Now find the If/Else block inside the loop
# We need to find `ElseIf stratName = "CUSTOM" Then` and similar, and add `ElseIf stratName = "MEAN" Then`
if_else_start = content.find('If stratName = "EQUAL WEIGHT" Then', loop_start_pos)
end_if_pos = content.find('End If', if_else_start)

# We want to insert the MEAN logic before the 'End If'
mean_logic = """
        ElseIf stratName = "MEAN" Then
            For m = 1 To nAssets
                optRange.Cells(1, m).Value = sumWeights(m) / wCount
            Next m
"""

# Let's see the current Else block
else_block = content.find('Else', if_else_start)
# We need to handle this carefully. The original structure is:
# If stratName = "EQUAL WEIGHT" Then
# ...
# ElseIf stratName = "CUSTOM" Then
# ...
# ElseIf stratName = "ER/VOL" Then
# ...
# Else
#   Call RunSolver(CStr(strategies(i)), minWeights, maxWeights, False)
# End If

# We can replace the 'Else' block to include MEAN, or replace 'End If' block.
# Let's do string replacement for the `Else` block:
old_else_block = """        Else

            ' Optimization: Sharpe, Variance, Kelly, ERC UNCSTRD
            Call RunSolver(CStr(strategies(i)), minWeights, maxWeights, False)

        End If

        Dim w As Variant
        w = Sheets(ENGINE_SHEET).Range("OptWeights").Value"""

new_else_block = """        ElseIf stratName = "MEAN" Then

            For m = 1 To nAssets
                optRange.Cells(1, m).Value = sumWeights(m) / wCount
            Next m

        Else

            ' Optimization: Sharpe, Variance, Kelly, ERC UNCSTRD
            Call RunSolver(CStr(strategies(i)), minWeights, maxWeights, False)

        End If

        Dim w As Variant
        w = Sheets(ENGINE_SHEET).Range("OptWeights").Value

        If stratName <> "MEAN" Then
            For m = 1 To nAssets
                sumWeights(m) = sumWeights(m) + w(1, m)
            Next m
        End If"""

content = content.replace(old_else_block, new_else_block)

with open('updated_vba_code.txt', 'w') as f:
    f.write(content)

print("Modifications applied successfully.")
