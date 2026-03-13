<template>
  <div class="w-full h-full">
    <div v-if="loaded" ref="chartContainer" class="w-full h-full"></div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch } from 'vue'
import { Chart, registerables } from 'chart.js'
import { annotationPlugin } from 'chartjs-plugin-annotation'

Chart.register(...registerables, annotationPlugin)

const props = defineProps({
  title: {
    type: String,
    default: ''
  },
  data: {
    type: Array,
    required: true
  },
  labels: {
    type: Array,
    required: true
  },
  showLegend: {
    type: Boolean,
    default: true
  },
  showTooltip: {
    type: Boolean,
    default: true
  },
  height: {
    type: Number,
    default: 400
  }
})

const loaded = ref(false)
const chartContainer = ref(null)
let chart = null

onMounted(() => {
  loaded.value = true
  createChart()
})

onUnmounted(() => {
  if (chart) {
    chart.destroy()
  }
})

watch(() => props.data, () => {
  if (chart) {
    updateChart()
  }
}, { deep: true })

function createChart() {
  if (!chartContainer.value) return

  const ctx = chartContainer.value.getContext('2d')
  
  const candleData = props.data.map((item, index) => ({
    x: props.labels[index],
    o: item.open,
    h: item.high,
    l: item.low,
    c: item.close
  }))

  chart = new Chart(ctx, {
    type: 'candlestick',
    data: {
      labels: props.labels,
      datasets: [{
        label: '股票价格',
        data: candleData,
        order: 1,
        color: {
          up: '#22c55e',
          down: '#ef4444',
          unchanged: '#9ca3af'
        },
        borderColor: {
          up: '#22c55e',
          down: '#ef4444',
          unchanged: '#9ca3af'
        },
        backgroundColor: {
          up: 'rgba(34, 197, 94, 0.5)',
          down: 'rgba(239, 68, 68, 0.5)',
          unchanged: 'rgba(156, 163, 175, 0.5)'
        }
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: {
          display: !!props.title,
          text: props.title,
          font: {
            size: 16,
            weight: 'bold'
          },
          padding: 20
        },
        legend: {
          display: props.showLegend,
          position: 'top'
        },
        tooltip: {
          enabled: props.showTooltip,
          mode: 'index',
          intersect: false,
          backgroundColor: 'rgba(0, 0, 0, 0.8)',
          titleFont: {
            size: 14
          },
          bodyFont: {
            size: 13
          },
          padding: 12,
          cornerRadius: 8,
          callbacks: {
            label: function(context) {
              const data = context.raw
              return [
                `开盘：${data.o}`,
                `最高：${data.h}`,
                `最低：${data.l}`,
                `收盘：${data.c}`
              ]
            }
          }
        }
      },
      scales: {
        x: {
          grid: {
            display: false
          },
          ticks: {
            maxRotation: 45,
            minRotation: 45
          }
        },
        y: {
          position: 'right',
          grid: {
            color: 'rgba(0, 0, 0, 0.05)'
          }
        }
      },
      interaction: {
        mode: 'nearest',
        axis: 'x',
        intersect: false
      }
    }
  })
}

function updateChart() {
  if (!chart) return

  const candleData = props.data.map((item, index) => ({
    x: props.labels[index],
    o: item.open,
    h: item.high,
    l: item.low,
    c: item.close
  }))

  chart.data.datasets[0].data = candleData
  chart.update()
}
</script>

<style scoped>
div {
  height: v-bind('height + "px"');
}
</style>
