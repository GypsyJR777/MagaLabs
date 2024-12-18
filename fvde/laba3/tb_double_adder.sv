// Testbench for IEEE Double Precision Floating Point Adder

`timescale 1ns/1ps

module tb_double_adder;

  // Inputs to the DUT
  logic clk;
  logic rst;
  logic [63:0] input_a;
  logic [63:0] input_b;
  logic input_a_stb;
  logic input_b_stb;
  logic output_z_ack;
  logic stb;

  // Outputs from the DUT
  logic [63:0] output_z;
  logic output_z_stb;
  logic input_a_ack;
  logic input_b_ack;

  // Variables
  logic [63:0] random_operand_a;
  logic [63:0] random_operand_b;
  logic [63:0] reference_result; // Declare reference_result for the DPI function
  int iteration_count = 0;       // Iteration counter
  real operand_a_real, operand_b_real, result_real;

  // Instantiate the DUT
  double_adder dut (
    .clk(clk),
    .rst(rst),
    .input_a(input_a),
    .input_b(input_b),
    .input_a_stb(stb),
    .input_b_stb(stb),
    .output_z_ack(output_z_ack),
    .output_z(output_z),
    .output_z_stb(output_z_stb),
    .input_a_ack(input_a_ack),
    .input_b_ack(input_b_ack)
  );

  // Import DPI function
  import "DPI-C" function void reference_adder(
    input longint a,
    input longint b,
    output longint result
  );

// Clock generation
  always #5 clk = ~clk;

// Сброс и начальная настройка
  initial begin
    clk = 0; rst = 1; stb = 0;
    #50 rst = 0;
  end

// Мониторинг всех сигналов
  // initial begin
  //   $monitor("Time=%0t | clk=%b rst=%b stb=%b input_a=%h input_b=%h output_z=%h output_z_stb=%b", 
  //            $time, clk, rst, stb, input_a, input_b, output_z, output_z_stb);
  // end
  
  // Test procedure
  initial begin
    // Initialize inputs
    rst = 1;
    clk = 0;
    stb = 0;
    // output_z_ack = 0;

    // Release reset after some time
    #50 rst = 0;

    // Run the test
     while (iteration_count < 13)  begin
      #10;
      // Generate random operands
      random_operand_a = {$random, $random};
      random_operand_b = {$random, $random};

      // Assign to inputs
      input_a = random_operand_a;
      input_b = random_operand_b;

      // Wait for DUT result
       // Устанавливаем строб-сигналы
    stb = 1;

   // Ожидание сигнала или таймаут
        fork
            begin
                wait (output_z_stb === 1'b1);
            end
            begin
                #1000;  // Таймаут на 500 нс
                $display("Timeout waiting for output_z_stb!");
            end
        join_any
        disable fork;
      // Display results
      $display("Operand A: %h, Operand B: %h", random_operand_a, random_operand_b);
      // Конвертация из 32-битного IEEE 754 в real
      operand_a_real = $bitstoreal({32'b0, random_operand_a}); // Для корректности double (64 бит)
      operand_b_real = $bitstoreal({32'b0, random_operand_b});
      result_real = $bitstoreal({32'b0, output_z});
        // Вывод чисел в формате double
      $display("Operand A: %e, Operand B: %e, Result: %e", operand_a_real, operand_b_real, result_real);

      // Call the reference function
      reference_adder(random_operand_a, random_operand_b, reference_result);

      // Display results
      // $display("Operand A: %h, Operand B: %h", random_operand_a, random_operand_b);
      $display("DUT Result: %h, Reference Result: %h", output_z, reference_result);

      // Check for mismatches
      if (output_z !== reference_result) begin
        $display("ERROR: Mismatch detected!");
      end else begin
        $display("Match: Results are correct.");
      end

      // Wait before the next iteration
      #20;
      stb = 0;
      iteration_count += 1;
      output_z_ack = 1;
      #10;
      output_z_ack = 0;
    end
  end

  // Clock generation

endmodule
